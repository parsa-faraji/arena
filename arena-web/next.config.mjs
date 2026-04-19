/** @type {import('next').NextConfig} */
const nextConfig = {
  // better-sqlite3 is a native module; don't try to bundle it for the server.
  serverExternalPackages: ["better-sqlite3"],
};

export default nextConfig;
