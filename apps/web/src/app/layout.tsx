import type { Metadata, Viewport } from 'next';
import { Outfit, JetBrains_Mono } from 'next/font/google';
import './globals.css';
import { AuthProvider } from '@/contexts/AuthContext';

// Premium typography - Outfit for headings, JetBrains Mono for code
const outfit = Outfit({
  subsets: ['latin'],
  variable: '--font-outfit',
  display: 'swap',
});

const jetbrainsMono = JetBrains_Mono({
  subsets: ['latin'],
  variable: '--font-mono',
  display: 'swap',
});

export const viewport: Viewport = {
  width: 'device-width',
  initialScale: 1,
  themeColor: '#FFFDFB',
};

export const metadata: Metadata = {
  title: 'ReBloom - AI Image Enhancement',
  description: 'Transform blurry photos into sharp, stunning images with AI-powered enhancement. 4x upscaling with Real-ESRGAN.',
  keywords: ['image enhancement', 'AI', 'deblur', 'upscale', 'photo restoration', 'Real-ESRGAN', 'image quality'],
  authors: [{ name: 'ReBloom' }],
  creator: 'ReBloom',
  publisher: 'ReBloom',
  robots: 'index, follow',
  openGraph: {
    title: 'ReBloom - AI Image Enhancement',
    description: 'Transform blurry photos into sharp, stunning images with AI',
    type: 'website',
    locale: 'en_US',
    siteName: 'ReBloom',
  },
  twitter: {
    card: 'summary_large_image',
    title: 'ReBloom - AI Image Enhancement',
    description: 'Transform blurry photos into sharp, stunning images',
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className={`${outfit.variable} ${jetbrainsMono.variable}`} suppressHydrationWarning>
      <body className="min-h-screen antialiased">
        <AuthProvider>{children}</AuthProvider>
      </body>
    </html>
  );
}
