import { createBundledHighlighter, createSingletonShorthands } from 'shiki/core';
import { createJavaScriptRegexEngine } from 'shiki/engine/javascript';

const languages = {
  astro: () => import('@shikijs/langs/astro'),
  bash: () => import('@shikijs/langs/bash'),
  css: () => import('@shikijs/langs/css'),
  diff: () => import('@shikijs/langs/diff'),
  dockerfile: () => import('@shikijs/langs/dockerfile'),
  html: () => import('@shikijs/langs/html'),
  ini: () => import('@shikijs/langs/ini'),
  javascript: () => import('@shikijs/langs/javascript'),
  json: () => import('@shikijs/langs/json'),
  jsonc: () => import('@shikijs/langs/jsonc'),
  jsx: () => import('@shikijs/langs/jsx'),
  markdown: () => import('@shikijs/langs/markdown'),
  mdx: () => import('@shikijs/langs/mdx'),
  scss: () => import('@shikijs/langs/scss'),
  svelte: () => import('@shikijs/langs/svelte'),
  toml: () => import('@shikijs/langs/toml'),
  tsx: () => import('@shikijs/langs/tsx'),
  typescript: () => import('@shikijs/langs/typescript'),
  vue: () => import('@shikijs/langs/vue'),
  xml: () => import('@shikijs/langs/xml'),
  yaml: () => import('@shikijs/langs/yaml'),
};

const themes = {
  'github-dark-default': () => import('@shikijs/themes/github-dark-default'),
  'github-light-default': () => import('@shikijs/themes/github-light-default'),
};

const createHighlighter = createBundledHighlighter({
  langs: languages,
  themes,
  engine: createJavaScriptRegexEngine,
});

const { codeToTokensWithThemes } = createSingletonShorthands(createHighlighter);

export type HighlightLanguage = keyof typeof languages;

export type HighlightedToken = {
  content: string;
  darkColor?: string;
  lightColor?: string;
  fontStyle?: number;
};

const extensionLanguages: Record<string, HighlightLanguage> = {
  astro: 'astro',
  bash: 'bash',
  cjs: 'javascript',
  css: 'css',
  htm: 'html',
  html: 'html',
  ini: 'ini',
  js: 'javascript',
  json: 'json',
  jsonc: 'jsonc',
  jsx: 'jsx',
  md: 'markdown',
  mdx: 'mdx',
  mjs: 'javascript',
  scss: 'scss',
  sh: 'bash',
  svelte: 'svelte',
  toml: 'toml',
  ts: 'typescript',
  tsx: 'tsx',
  vue: 'vue',
  xml: 'xml',
  yaml: 'yaml',
  yml: 'yaml',
  zsh: 'bash',
};

export function languageForPath(path: string): HighlightLanguage | null {
  const filename = path.split('/').pop()?.toLowerCase() ?? '';
  if (filename === 'dockerfile') return 'dockerfile';
  if (filename === 'bun.lock') return 'toml';
  if (filename === '.env' || filename.startsWith('.env.')) return 'bash';
  const extension = filename.split('.').pop() ?? '';
  return extensionLanguages[extension] ?? null;
}

export async function highlightSource(
  source: string,
  language: HighlightLanguage,
): Promise<HighlightedToken[][]> {
  const tokenLines = await codeToTokensWithThemes(source, {
    lang: language,
    themes: {
      dark: 'github-dark-default',
      light: 'github-light-default',
    },
  });

  return tokenLines.map((line) => line.map((token) => ({
    content: token.content,
    darkColor: token.variants.dark?.color,
    lightColor: token.variants.light?.color,
    fontStyle: token.variants.dark?.fontStyle ?? token.variants.light?.fontStyle,
  })));
}
