import type { Metadata } from 'next';

export const metadata: Metadata = {
  title: 'ReBloom - AI Image Enhancement',
  description: 'Transform blurry photos into sharp, stunning images with AI-powered enhancement.',
};

// Root layout - minimal, just provides html structure
// Actual layout with providers is in [locale]/layout.tsx
export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return children;
}
