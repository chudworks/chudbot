import fs from "node:fs";
import path from "node:path";
import process from "node:process";
import { TextDecoder } from "node:util";

const ROOT = "/workspace";
const RESERVED = new Set([".git", "node_modules", "dist"]);

function fail(message) {
  process.stdout.write(JSON.stringify({ ok: false, error: message }));
  process.exit(0);
}

function relativePath(value, allowRoot = false) {
  if (typeof value !== "string" || value.includes("\\") || value.includes("\0")) {
    fail("invalid workspace path");
  }
  if (value === "" && allowRoot) return "";
  const parts = value.split("/");
  if (
    parts.length === 0 ||
    parts.some((part) => part === "" || part === "." || part === ".." || RESERVED.has(part))
  ) {
    fail("invalid workspace path");
  }
  return parts.join("/");
}

function inspectExisting(relative) {
  let current = ROOT;
  if (relative === "") return current;
  for (const part of relative.split("/")) {
    current = path.posix.join(current, part);
    try {
      const metadata = fs.lstatSync(current);
      if (metadata.isSymbolicLink()) fail("symbolic links are not allowed");
    } catch (error) {
      if (error?.code === "ENOENT") break;
      throw error;
    }
  }
  return path.posix.join(ROOT, relative);
}

function requireOrdinaryFile(target) {
  const metadata = fs.lstatSync(target);
  if (metadata.isSymbolicLink() || !metadata.isFile()) fail("path is not an ordinary file");
  return metadata;
}

function decodeUtf8(bytes) {
  try {
    return new TextDecoder("utf-8", { fatal: true }).decode(bytes);
  } catch {
    fail("file is not UTF-8 text");
  }
}

function atomicWrite(target, contents) {
  const temporary = path.posix.join(
    path.posix.dirname(target),
    `.vibe-edit-${process.pid}-${Date.now()}-${Math.random().toString(16).slice(2)}`,
  );
  try {
    fs.writeFileSync(temporary, contents, { flag: "wx" });
    fs.renameSync(temporary, target);
  } finally {
    try {
      fs.unlinkSync(temporary);
    } catch (error) {
      if (error?.code !== "ENOENT") throw error;
    }
  }
}

let request;
try {
  const chunks = [];
  for await (const chunk of process.stdin) chunks.push(chunk);
  request = JSON.parse(Buffer.concat(chunks).toString("utf8"));
} catch {
  fail("invalid file tool request");
}

try {
  const operation = request?.operation;
  const relative = relativePath(request?.path, operation === "read");
  const target = inspectExisting(relative);

  if (operation === "read") {
    const metadata = fs.lstatSync(target);
    if (metadata.isSymbolicLink()) fail("symbolic links are not allowed");
    if (metadata.isDirectory()) {
      const entries = fs
        .readdirSync(target)
        .filter((name) => !RESERVED.has(name))
        .sort();
      process.stdout.write(JSON.stringify({ ok: true, kind: "directory", entries }));
    } else if (metadata.isFile()) {
      if (metadata.size > request.maxBytes) fail("file is too large for read");
      const bytes = fs.readFileSync(target);
      process.stdout.write(
        JSON.stringify({ ok: true, kind: "file", data: bytes.toString("base64") }),
      );
    } else {
      fail("path is not an ordinary file or directory");
    }
  } else if (operation === "create") {
    if (typeof request.content !== "string") fail("`content` must be a string");
    if (fs.existsSync(target)) fail("file already exists; reread it and use replace");
    fs.mkdirSync(path.posix.dirname(target), { recursive: true });
    fs.writeFileSync(target, request.content, { flag: "wx" });
    process.stdout.write(JSON.stringify({ ok: true }));
  } else if (operation === "delete") {
    requireOrdinaryFile(target);
    fs.unlinkSync(target);
    process.stdout.write(JSON.stringify({ ok: true }));
  } else if (operation === "replace") {
    if (typeof request.old !== "string" || request.old === "") fail("old must not be empty");
    if (typeof request.new !== "string") fail("`new` must be a string");
    requireOrdinaryFile(target);
    const text = decodeUtf8(fs.readFileSync(target));
    const count = text.split(request.old).length - 1;
    if (count !== 1) {
      fail(`replacement matched ${count} times; reread the file and provide one exact unique string`);
    }
    atomicWrite(target, text.replace(request.old, request.new));
    process.stdout.write(JSON.stringify({ ok: true }));
  } else {
    fail("unknown file operation");
  }
} catch (error) {
  if (error?.code === "ENOENT") fail("file does not exist; reread the directory");
  fail("workspace file operation failed");
}
