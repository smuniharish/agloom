#!/bin/sh
set -eu
: "${EVIDENCE_BROWSER_PROXY:?Evidence browser proxy must be running}"
exec /usr/local/bin/agloom-chromium-real "$@" \
  --proxy-server="$EVIDENCE_BROWSER_PROXY" \
  --proxy-bypass-list="<-loopback>" \
  --disable-quic \
  --force-webrtc-ip-handling-policy=disable_non_proxied_udp
