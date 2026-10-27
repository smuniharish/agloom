FROM node:24-bookworm-slim

ENV PLAYWRIGHT_BROWSERS_PATH=/ms-playwright
WORKDIR /opt/browser
COPY examples/continuous_evidence_lab/browser-service/package.json \
     examples/continuous_evidence_lab/browser-service/package-lock.json ./
RUN apt-get update \
 && apt-get install -y --no-install-recommends python3 \
 && rm -rf /var/lib/apt/lists/* \
 && npm ci --ignore-scripts --no-audit --no-fund
RUN npx playwright install --with-deps chromium
RUN test -x /ms-playwright/chromium-1247/chrome-linux64/chrome \
 && actual_version=$(/ms-playwright/chromium-1247/chrome-linux64/chrome --version \
    | sed 's/[[:space:]]*$//') \
 && echo "Verifying patched browser: ${actual_version}" \
 && test "${actual_version}" = "Google Chrome for Testing 155.0.8059.12" \
 && ln -s /ms-playwright/chromium-1247/chrome-linux64/chrome \
    /usr/local/bin/agloom-chromium-real \
 && useradd --create-home --uid 10001 browser
COPY examples/continuous_evidence_lab/backend/browser_proxy.py /opt/browser/browser_proxy.py
COPY examples/continuous_evidence_lab/browser-chromium.sh /usr/local/bin/agloom-chromium
RUN chmod 0555 /usr/local/bin/agloom-chromium /opt/browser/browser_proxy.py \
 && chown -R browser:browser /opt/browser
USER browser
EXPOSE 8931
CMD ["python3", "/opt/browser/browser_proxy.py"]
