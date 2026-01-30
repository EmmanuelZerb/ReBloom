import type { Metadata, Viewport } from 'next';
import { Outfit, JetBrains_Mono } from 'next/font/google';
import { notFound } from 'next/navigation';
import { NextIntlClientProvider } from 'next-intl';
import { getMessages, setRequestLocale } from 'next-intl/server';
import { AuthProvider } from '@/contexts/AuthContext';
import { ErrorBoundary } from '@/components/errors/ErrorBoundary';
import { locales, type Locale } from '@/i18n/config';
import '../globals.css';

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
  themeColor: '#FFFFFF',
  colorScheme: 'light',
};

export function generateStaticParams() {
  return locales.map((locale) => ({ locale }));
}

type Props = {
  children: React.ReactNode;
  params: Promise<{ locale: string }>;
};

export async function generateMetadata({ params }: Props): Promise<Metadata> {
  const { locale } = await params;

  const titles: Record<string, string> = {
    en: 'ReBloom - AI Image Enhancement',
    fr: 'ReBloom - Amélioration d\'images par IA',
    de: 'ReBloom - KI-Bildverbesserung',
    es: 'ReBloom - Mejora de imágenes con IA',
    it: 'ReBloom - Miglioramento immagini con IA',
    pt: 'ReBloom - Melhoria de imagens com IA',
    nl: 'ReBloom - AI-beeldverbetering',
    pl: 'ReBloom - Ulepszanie obrazów AI',
    sv: 'ReBloom - AI-bildförbättring',
    da: 'ReBloom - AI-billedforbedring',
    no: 'ReBloom - AI-bildeforbedring',
    fi: 'ReBloom - Tekoälykuvanparannus',
  };

  const descriptions: Record<string, string> = {
    en: 'Transform blurry photos into sharp, stunning images with AI-powered enhancement.',
    fr: 'Transformez vos photos floues en images nettes grâce à l\'amélioration par IA.',
    de: 'Verwandeln Sie unscharfe Fotos mit KI-gestützter Verbesserung in scharfe Bilder.',
    es: 'Transforma fotos borrosas en imágenes nítidas con mejora impulsada por IA.',
    it: 'Trasforma le foto sfocate in immagini nitide con il miglioramento basato sull\'IA.',
    pt: 'Transforme fotos borradas em imagens nítidas com melhoria por IA.',
    nl: 'Transformeer wazige foto\'s in scherpe beelden met AI-verbetering.',
    pl: 'Przekształć rozmyte zdjęcia w ostre obrazy dzięki ulepszaniu AI.',
    sv: 'Förvandla suddiga foton till skarpa bilder med AI-förbättring.',
    da: 'Forvandl slørede fotos til skarpe billeder med AI-forbedring.',
    no: 'Forvandle uskarpe bilder til skarpe bilder med AI-forbedring.',
    fi: 'Muuta sumeat kuvat teräväksi tekoälyparannuksella.',
  };

  return {
    title: titles[locale] || titles.en,
    description: descriptions[locale] || descriptions.en,
    keywords: ['image enhancement', 'AI', 'deblur', 'upscale', 'photo restoration', 'Real-ESRGAN'],
    authors: [{ name: 'ReBloom' }],
    creator: 'ReBloom',
    publisher: 'ReBloom',
    robots: 'index, follow',
    metadataBase: new URL(process.env.NEXT_PUBLIC_APP_URL || 'http://localhost:3000'),
    alternates: {
      canonical: `/${locale}`,
      languages: Object.fromEntries(locales.map((l) => [l, `/${l}`])),
    },
    openGraph: {
      title: titles[locale] || titles.en,
      description: descriptions[locale] || descriptions.en,
      type: 'website',
      locale: locale,
      siteName: 'ReBloom',
    },
    twitter: {
      card: 'summary_large_image',
      title: titles[locale] || titles.en,
      description: descriptions[locale] || descriptions.en,
    },
  };
}

export default async function LocaleLayout({ children, params }: Props) {
  const { locale } = await params;

  // Validate locale
  if (!locales.includes(locale as Locale)) {
    notFound();
  }

  // Enable static rendering
  setRequestLocale(locale);

  // Get messages for the current locale
  const messages = await getMessages();

  return (
    <html lang={locale} className={`${outfit.variable} ${jetbrainsMono.variable}`} suppressHydrationWarning>
      <body className="min-h-screen antialiased">
        <NextIntlClientProvider messages={messages}>
          <ErrorBoundary>
            <AuthProvider>{children}</AuthProvider>
          </ErrorBoundary>
        </NextIntlClientProvider>
      </body>
    </html>
  );
}
