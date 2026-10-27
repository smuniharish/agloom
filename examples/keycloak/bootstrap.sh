#!/bin/sh
set -eu

: "${KEYCLOAK_ADMIN:?KEYCLOAK_ADMIN is required}"
: "${KEYCLOAK_ADMIN_PASSWORD:?KEYCLOAK_ADMIN_PASSWORD is required}"
: "${OIDC_DEMO_USERNAME:?OIDC_DEMO_USERNAME is required}"
: "${OIDC_DEMO_PASSWORD:?OIDC_DEMO_PASSWORD is required}"
: "${OIDC_DEMO_TENANT:?OIDC_DEMO_TENANT is required}"

KCADM=/opt/keycloak/bin/kcadm.sh
until "$KCADM" config credentials --server http://keycloak:8080 \
  --realm master --user "$KEYCLOAK_ADMIN" --password "$KEYCLOAK_ADMIN_PASSWORD"; do
  sleep 2
done

"$KCADM" get realms/agloom >/dev/null 2>&1 ||
  "$KCADM" create realms -s realm=agloom -s enabled=true
"$KCADM" update users/profile -r agloom -s unmanagedAttributePolicy=ADMIN_EDIT

create_client() {
  client="$1"
  redirect="$2"
  existing=$("$KCADM" get clients -r agloom -q clientId="$client" --fields id --format csv --noquotes 2>/dev/null || true)
  if [ -z "$existing" ]; then
    "$KCADM" create clients -r agloom -s clientId="$client" -s enabled=true \
      -s publicClient=true -s standardFlowEnabled=true -s directAccessGrantsEnabled=false \
      -s "webOrigins=[\"$redirect\"]" -s "redirectUris=[\"$redirect/*\"]"
    existing=$("$KCADM" get clients -r agloom -q clientId="$client" --fields id --format csv --noquotes)
  fi
  mapper=$("$KCADM" get "clients/$existing/protocol-mappers/models" -r agloom 2>/dev/null | grep -c '"name" : "tenant_id"' || true)
  if [ "$mapper" -eq 0 ]; then
    "$KCADM" create "clients/$existing/protocol-mappers/models" -r agloom \
      -s name=tenant_id -s protocol=openid-connect \
      -s protocolMapper=oidc-usermodel-attribute-mapper \
      -s 'config."user.attribute"=tenant_id' -s 'config."claim.name"=tenant_id' \
      -s 'config."jsonType.label"=String' -s 'config."access.token.claim"=true'
  fi
  audience=$("$KCADM" get "clients/$existing/protocol-mappers/models" -r agloom 2>/dev/null | grep -c '"name" : "audience"' || true)
  if [ "$audience" -eq 0 ]; then
    "$KCADM" create "clients/$existing/protocol-mappers/models" -r agloom \
      -s name=audience -s protocol=openid-connect \
      -s protocolMapper=oidc-audience-mapper \
      -s "config.\"included.client.audience\"=$client" \
      -s 'config."access.token.claim"=true'
  fi
}

create_client product-studio http://localhost:5178
create_client evidence-lab http://localhost:5180

user_id=$("$KCADM" get users -r agloom -q username="$OIDC_DEMO_USERNAME" --fields id --format csv --noquotes 2>/dev/null || true)
if [ -z "$user_id" ]; then
  user_id=$("$KCADM" create users -r agloom -s username="$OIDC_DEMO_USERNAME" \
    -s enabled=true -s firstName=Release -s lastName=Demo \
    -s "email=$OIDC_DEMO_USERNAME@agloom.local" -s emailVerified=true \
    -s "attributes.tenant_id=[\"$OIDC_DEMO_TENANT\"]" -i)
else
  "$KCADM" update "users/$user_id" -r agloom \
    -s firstName=Release -s lastName=Demo \
    -s "email=$OIDC_DEMO_USERNAME@agloom.local" -s emailVerified=true \
    -s "attributes.tenant_id=[\"$OIDC_DEMO_TENANT\"]"
fi
"$KCADM" set-password -r agloom --userid "$user_id" --new-password "$OIDC_DEMO_PASSWORD"
