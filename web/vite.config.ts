import { constants as zc } from "node:zlib";
import { fileURLToPath } from "node:url";
import { compression, defineAlgorithm } from "vite-plugin-compression2";
// `vitest/config` re-exports vite's defineConfig with the `test` key typed.
import { defineConfig } from "vitest/config";

// TECH_PLAN.md §5.2 / §5.3:
//   - precompress at BUILD time (brotli-11 + gzip siblings). Never at request time.
//   - content-hashed filenames, served `immutable` for a year.
//   - assetsInlineLimit 0: the sampling table and the tokenizer blob must stay
//     separate, content-hashed, cacheable files. A base64 data: URI would push
//     8 KB of key material into the JS bundle and blow the 60,000 B gate.
export default defineConfig({
  root: fileURLToPath(new URL(".", import.meta.url)),
  publicDir: false,
  resolve: {
    alias: {
      "@vendor/tokenizers": fileURLToPath(
        new URL("./vendor/hf-tokenizers-0.1.3.mjs", import.meta.url),
      ),
    },
  },
  worker: {
    format: "es",
  },
  build: {
    target: "es2022",
    assetsInlineLimit: 0,
    cssCodeSplit: false,
    sourcemap: false,
    reportCompressedSize: false,
    rollupOptions: {
      output: {
        entryFileNames: "assets/[name]-[hash].js",
        chunkFileNames: "assets/[name]-[hash].js",
        assetFileNames: "assets/[name]-[hash][extname]",
      },
    },
  },
  plugins: [
    compression({
      // brotli-11 + gzip-9 siblings, emitted alongside the originals so the
      // server can negotiate. `deleteOriginalAssets: false` because a client
      // with no `Accept-Encoding` still has to be served something.
      algorithms: [
        defineAlgorithm("brotliCompress", {
          params: {
            [zc.BROTLI_PARAM_QUALITY]: 11,
            [zc.BROTLI_PARAM_LGWIN]: 24,
          },
        }),
        defineAlgorithm("gzip", { level: 9 }),
      ],
      exclude: [/\.br$/, /\.gz$/],
      deleteOriginalAssets: false,
      threshold: 0,
    }),
  ],
  test: {
    include: ["tests/**/*.test.ts"],
    environment: "node",
    // The tokenizer parity suite builds two full 262,144-piece tokenizers and
    // runs 4,031 golden vectors through each. That is minutes, not seconds.
    testTimeout: 600_000,
    hookTimeout: 600_000,
  },
});
