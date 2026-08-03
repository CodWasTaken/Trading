import type { Metadata } from "next";
import Link from "next/link";
import "./globals.css";

export const metadata: Metadata = {
  title: "Trading Control Room",
  description: "Governed multi-model paper trading dashboard",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>
        <nav
          aria-label="Primary"
          style={{
            display: "flex",
            gap: "1rem",
            padding: "1rem max(1rem, calc((100vw - 1400px) / 2))",
            borderBottom: "1px solid rgba(148, 163, 184, 0.18)",
          }}
        >
          <Link href="/">Control room</Link>
          <Link href="/models">Models</Link>
          <span style={{ marginLeft: "auto", opacity: 0.72 }}>Paper only</span>
        </nav>
        {children}
      </body>
    </html>
  );
}
