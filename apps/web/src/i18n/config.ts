export const locales = ['en', 'fr', 'de', 'es', 'it', 'pt', 'nl', 'pl', 'sv', 'da', 'no', 'fi'] as const;
export type Locale = (typeof locales)[number];

export const defaultLocale: Locale = 'en';

export const localeNames: Record<Locale, string> = {
  en: 'English',
  fr: 'Français',
  de: 'Deutsch',
  es: 'Español',
  it: 'Italiano',
  pt: 'Português',
  nl: 'Nederlands',
  pl: 'Polski',
  sv: 'Svenska',
  da: 'Dansk',
  no: 'Norsk',
  fi: 'Suomi',
};

export const localeCodes: Record<Locale, string> = {
  en: 'EN',
  fr: 'FR',
  de: 'DE',
  es: 'ES',
  it: 'IT',
  pt: 'PT',
  nl: 'NL',
  pl: 'PL',
  sv: 'SV',
  da: 'DA',
  no: 'NO',
  fi: 'FI',
};

// Mapping country codes to locales
export const countryToLocale: Record<string, Locale> = {
  // French
  FR: 'fr', BE: 'fr', CH: 'fr', CA: 'fr', LU: 'fr', MC: 'fr',
  // German
  DE: 'de', AT: 'de', LI: 'de',
  // Spanish
  ES: 'es', AR: 'es', MX: 'es', CO: 'es', CL: 'es', PE: 'es',
  // Italian
  IT: 'it', SM: 'it', VA: 'it',
  // Portuguese
  PT: 'pt', BR: 'pt',
  // Dutch
  NL: 'nl',
  // Polish
  PL: 'pl',
  // Swedish
  SE: 'sv',
  // Danish
  DK: 'da',
  // Norwegian
  NO: 'no',
  // Finnish
  FI: 'fi',
  // English (default)
  GB: 'en', US: 'en', AU: 'en', NZ: 'en', IE: 'en',
};

export function getLocaleFromCountry(countryCode: string | null): Locale {
  if (!countryCode) return defaultLocale;
  return countryToLocale[countryCode.toUpperCase()] || defaultLocale;
}
