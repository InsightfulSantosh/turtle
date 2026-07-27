import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Turtle Season Intelligence | AI Merchandise Planning",
  description:
    "FashionCLIP product matching, validated demand modelling, uncertainty ranges, and explainable seasonal order recommendations.",
  icons: { icon: "/favicon.png", shortcut: "/favicon.png" },
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
