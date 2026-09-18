const { defineConfig } = require('@playwright/test');
module.exports = defineConfig({
  testDir: './tests/browser',
  use: { baseURL: 'http://127.0.0.1:8765', browserName: 'chromium' },
  webServer: { command: 'python3 -m http.server 8765 --bind 127.0.0.1 --directory docs', url: 'http://127.0.0.1:8765', reuseExistingServer: false },
});
