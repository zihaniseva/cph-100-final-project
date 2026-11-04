const { chromium } = require('playwright');
const path = require('path');
(async () => {
  // usage: node render.js <src dir> <out dir> task pipeline structures splits patching timeline
  const [dir, out, ...names] = process.argv.slice(2);
  const b = await chromium.launch();
  const p = await b.newPage({ viewport: { width: 1200, height: 900 }, deviceScaleFactor: 2 });
  for (const n of names) {
    await p.goto('file://' + path.join(dir, n + '.html'));
    await p.evaluate(() => document.fonts.ready);
    await p.locator('.figure').screenshot({ path: path.join(out, n + '.png') });
    console.log('ok', n);
  }
  await b.close();
})();
