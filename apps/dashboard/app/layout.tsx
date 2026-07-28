import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Trading Control Room",
  description: "Live AI-assisted paper trading dashboard",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
