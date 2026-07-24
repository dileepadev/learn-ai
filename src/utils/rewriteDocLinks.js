import path from 'node:path';

export const SKIP_REWRITE_LINK_PATTERN = /^(?:[a-z]+:|\/\/|\/|#|\?)/i;
export const DOC_PAGE_EXTENSION_PATTERN = /\.(md|mdx)$/;
export const DOC_LINK_PARTS_PATTERN = /^([^?#]+)(\?[^#]*)?(#.*)?$/;

export function getDocDirectory(filePath, docsRoot) {
  if (!filePath?.startsWith(docsRoot)) {
    return null;
  }

  const relativeFilePath = path
    .relative(docsRoot, filePath)
    .split(path.sep)
    .join(path.posix.sep);

  return path.posix.dirname(relativeFilePath);
}

export function rewriteDocLink(url, currentDir, basePath) {
  const trimmedUrl = url.trim();

  if (SKIP_REWRITE_LINK_PATTERN.test(trimmedUrl)) {
    return trimmedUrl;
  }

  // Capture groups: pathname before "?" or "#", optional query string, optional hash fragment.
  const match = trimmedUrl.match(DOC_LINK_PARTS_PATTERN);

  if (!match) {
    return trimmedUrl;
  }

  const [, pathname, search = '', hash = ''] = match;
  const resolvedPath = path.posix
    .join('/', currentDir, pathname)
    .replace(DOC_PAGE_EXTENSION_PATTERN, '')
    .replace(/\/index$/, '/');
  const normalizedBasePath = basePath.replace(/\/$/, '');

  return `${normalizedBasePath}${resolvedPath}${search}${hash}`;
}
