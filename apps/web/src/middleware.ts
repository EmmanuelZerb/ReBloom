/**
 * Next.js Middleware
 * Handles i18n locale detection and Supabase session refresh
 */

import { type NextRequest, NextResponse } from 'next/server';
import createMiddleware from 'next-intl/middleware';
import { updateSession } from '@/lib/supabase/middleware';
import { routing } from '@/i18n/navigation';
import { getLocaleFromCountry, locales, defaultLocale } from '@/i18n/config';

const intlMiddleware = createMiddleware(routing);

export async function middleware(request: NextRequest) {
  const { pathname } = request.nextUrl;

  // Skip locale handling for API routes and static files
  if (
    pathname.startsWith('/api') ||
    pathname.startsWith('/_next') ||
    pathname.includes('.')
  ) {
    return await updateSession(request);
  }

  // Check for saved locale preference in cookie
  const savedLocale = request.cookies.get('NEXT_LOCALE')?.value;

  // If no saved preference, detect from Vercel Geolocation
  if (!savedLocale) {
    const country = request.headers.get('x-vercel-ip-country');
    const detectedLocale = getLocaleFromCountry(country);

    // If detected locale is different from default, redirect
    if (detectedLocale !== defaultLocale) {
      const pathWithoutLocale = pathname.replace(/^\/[a-z]{2}(?=\/|$)/, '') || '/';
      const newUrl = new URL(`/${detectedLocale}${pathWithoutLocale}`, request.url);

      const response = NextResponse.redirect(newUrl);
      response.cookies.set('NEXT_LOCALE', detectedLocale, {
        maxAge: 60 * 60 * 24 * 365, // 1 year
        path: '/',
      });
      return response;
    }
  }

  // Run next-intl middleware
  const intlResponse = intlMiddleware(request);

  // Run Supabase session update
  const sessionResponse = await updateSession(request);

  // Merge headers from both responses
  if (intlResponse) {
    sessionResponse.headers.forEach((value, key) => {
      intlResponse.headers.set(key, value);
    });
    return intlResponse;
  }

  return sessionResponse;
}

export const config = {
  matcher: [
    /*
     * Match all request paths except for the ones starting with:
     * - _next/static (static files)
     * - _next/image (image optimization files)
     * - favicon.ico (favicon file)
     * - public folder
     */
    '/((?!_next/static|_next/image|favicon.ico|.*\\.(?:svg|png|jpg|jpeg|gif|webp)$).*)',
  ],
};
