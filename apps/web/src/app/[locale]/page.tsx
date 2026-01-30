'use client';

import { memo, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Check, Download, RefreshCw, User, Sparkles } from 'lucide-react';
import { useTranslations } from 'next-intl';
import { Link } from '@/i18n/navigation';
import { DropZone } from '@/components/upload/DropZone';
import { ProcessingStatus } from '@/components/progress/ProcessingStatus';
import { ImageCompare } from '@/components/compare/ImageCompare';
import { LanguageSwitcher } from '@/components/LanguageSwitcher';
import { useUpload } from '@/hooks/useUpload';
import { useAuth } from '@/contexts/AuthContext';

const FeatureBadge = memo(function FeatureBadge({
  children
}: {
  children: React.ReactNode
}) {
  return (
    <span className="flex items-center gap-2">
      <span className="w-5 h-5 rounded-full bg-sage-50 flex items-center justify-center">
        <Check className="w-3 h-3 text-sage" aria-hidden="true" />
      </span>
      {children}
    </span>
  );
});

const StepCard = memo(function StepCard({
  num,
  title,
  desc,
  delay
}: {
  num: string;
  title: string;
  desc: string;
  delay: number;
}) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true }}
      transition={{ delay }}
      className="relative group"
    >
      <span className="inline-flex items-center justify-center w-10 h-10 rounded-xl bg-coral-50 text-coral font-mono text-sm mb-4 group-hover:scale-110 transition-transform">
        {num}
      </span>
      <h3 className="text-xl font-medium text-stone-800 mb-2">
        {title}
      </h3>
      <p className="text-stone-400 leading-relaxed">
        {desc}
      </p>
    </motion.div>
  );
});

export default function Home() {
  const { state, upload, reset, previewUrl } = useUpload();
  const { user, signOut, isLoading } = useAuth();
  const t = useTranslations();

  const steps = useMemo(() => [
    {
      num: '01',
      title: t('home.howItWorks.step1.title'),
      desc: t('home.howItWorks.step1.desc')
    },
    {
      num: '02',
      title: t('home.howItWorks.step2.title'),
      desc: t('home.howItWorks.step2.desc')
    },
    {
      num: '03',
      title: t('home.howItWorks.step3.title'),
      desc: t('home.howItWorks.step3.desc')
    },
  ], [t]);

  return (
    <main className="min-h-screen bg-white relative overflow-hidden">
      <a
        href="#main-content"
        className="sr-only focus:not-sr-only focus:absolute focus:top-4 focus:left-4 focus:z-50 focus:px-4 focus:py-2 focus:bg-stone-900 focus:text-white focus:rounded-lg"
      >
        {t('nav.skipToContent')}
      </a>

      <div
        className="absolute inset-0 pointer-events-none"
        style={{
          background: 'radial-gradient(ellipse 80% 50% at 50% -20%, rgba(255, 127, 102, 0.08), transparent)',
        }}
        aria-hidden="true"
      />

      {/* Header */}
      <header className="relative px-6 py-6 md:px-12">
        <nav className="max-w-6xl mx-auto flex items-center justify-between" role="navigation" aria-label="Main navigation">
          <motion.a
            href="/"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="text-lg font-semibold text-stone-900 tracking-tight flex items-center gap-2"
            aria-label="ReBloom - Home"
          >
            <Sparkles className="w-5 h-5 text-coral" aria-hidden="true" />
            ReBloom
          </motion.a>

          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.1 }}
            className="flex items-center gap-6"
          >
            <a
              href="#how"
              className="text-sm text-stone-500 hover:text-stone-900 transition-colors hidden sm:block"
            >
              {t('nav.howItWorks')}
            </a>
            <a
              href="https://github.com/your-repo/rebloom"
              target="_blank"
              rel="noopener noreferrer"
              className="text-sm text-stone-500 hover:text-stone-900 transition-colors hidden sm:block"
            >
              {t('nav.github')}
            </a>

            <LanguageSwitcher />

            {!isLoading && (
              user ? (
                <div className="flex items-center gap-4">
                  <span className="text-sm text-stone-500 hidden sm:block">
                    {user.email?.split('@')[0]}
                  </span>
                  <button
                    onClick={() => signOut()}
                    className="text-sm text-stone-400 hover:text-stone-900 transition-colors"
                  >
                    {t('nav.signOut')}
                  </button>
                </div>
              ) : (
                <Link
                  href="/auth/login"
                  className="text-sm font-medium text-white bg-stone-900 hover:bg-stone-800 px-4 py-2 rounded-full transition-colors flex items-center gap-2"
                >
                  <User className="w-3.5 h-3.5" aria-hidden="true" />
                  {t('nav.signIn')}
                </Link>
              )
            )}
          </motion.div>
        </nav>
      </header>

      {/* Hero */}
      <section className="relative px-6 md:px-12 pt-12 md:pt-24 pb-20" aria-labelledby="hero-heading">
        <div className="max-w-6xl mx-auto">
          <div className="max-w-2xl">
            <motion.h1
              id="hero-heading"
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.05 }}
              className="text-4xl md:text-6xl font-semibold text-stone-900 leading-[1.1] tracking-tight mb-6"
            >
              {t('home.title')}{' '}
              <span className="relative">
                {t('home.titleHighlight')}
                <svg
                  className="absolute -bottom-2 left-0 w-full"
                  viewBox="0 0 200 8"
                  fill="none"
                  aria-hidden="true"
                >
                  <path
                    d="M2 6c50-4 100-4 196 0"
                    stroke="#FF7F66"
                    strokeWidth="3"
                    strokeLinecap="round"
                    className="opacity-40"
                  />
                </svg>
              </span>
            </motion.h1>

            <motion.p
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.1 }}
              className="text-lg text-stone-500 leading-relaxed mb-8"
            >
              {t('home.subtitle')}
            </motion.p>

            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.15 }}
              className="flex flex-wrap items-center gap-5 text-sm text-stone-400"
              role="list"
              aria-label="Key features"
            >
              <FeatureBadge>{t('home.features.free')}</FeatureBadge>
              <FeatureBadge>{t('home.features.upscaling')}</FeatureBadge>
              <FeatureBadge>{t('home.features.private')}</FeatureBadge>
            </motion.div>
          </div>
        </div>
      </section>

      {/* Main App Area */}
      <section id="main-content" className="relative px-6 md:px-12 pb-32" aria-label="Image upload and processing">
        <div className="max-w-6xl mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
          >
            <AnimatePresence mode="wait">
              {state.status === 'idle' && (
                <motion.div
                  key="idle"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0, y: -10 }}
                >
                  <DropZone onFileSelect={(file) => upload(file)} />
                </motion.div>
              )}

              {state.status === 'uploading' && (
                <motion.div
                  key="uploading"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  className="card p-12 text-center max-w-xl mx-auto"
                  role="status"
                  aria-live="polite"
                >
                  <div className="w-12 h-12 mx-auto mb-6 rounded-full border-2 border-stone-200 border-t-stone-900 animate-spin" aria-hidden="true" />
                  <p className="text-stone-500">{t('upload.uploading')}</p>
                </motion.div>
              )}

              {state.status === 'processing' && (
                <motion.div
                  key="processing"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  className="max-w-xl mx-auto space-y-6"
                  role="status"
                  aria-live="polite"
                >
                  {previewUrl && (
                    <div className="card p-4">
                      <div className="relative aspect-video rounded-xl overflow-hidden bg-stone-100">
                        <img
                          src={previewUrl}
                          alt="Image being processed"
                          className="w-full h-full object-contain opacity-60"
                          loading="lazy"
                        />
                        <div className="absolute inset-0 flex items-center justify-center">
                          <span className="px-3 py-1.5 rounded-full bg-white/90 text-sm text-stone-600 shadow-soft">
                            {t('upload.processing')}
                          </span>
                        </div>
                      </div>
                    </div>
                  )}
                  <ProcessingStatus job={state.job} />
                </motion.div>
              )}

              {state.status === 'completed' && (
                <motion.div
                  key="completed"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  className="space-y-8"
                >
                  {previewUrl && state.job.processedImage && (
                    <ImageCompare
                      originalUrl={previewUrl}
                      enhancedUrl={state.job.processedImage.url}
                    />
                  )}

                  <div className="max-w-xl mx-auto">
                    <div className="card p-6">
                      <div className="flex items-center justify-between mb-6">
                        <div>
                          <h3 className="font-medium text-stone-900">{t('result.ready')}</h3>
                          <p className="text-sm text-stone-500">
                            {state.job.processedImage && (
                              <>
                                {state.job.processedImage.width} × {state.job.processedImage.height}px
                                {state.job.metadata.processingTimeMs && (
                                  <span className="ml-2">
                                    · {(state.job.metadata.processingTimeMs / 1000).toFixed(1)}s
                                  </span>
                                )}
                              </>
                            )}
                          </p>
                        </div>
                        <div className="w-10 h-10 rounded-full bg-sage-50 flex items-center justify-center">
                          <Check className="w-5 h-5 text-sage" aria-hidden="true" />
                        </div>
                      </div>

                      <div className="flex items-center gap-3">
                        <a
                          href={state.downloadUrl}
                          download={state.job.processedImage?.originalName || 'enhanced.png'}
                          className="btn-prism flex-1 flex items-center justify-center gap-2"
                        >
                          <Download className="w-4 h-4" aria-hidden="true" />
                          {t('result.download')}
                        </a>
                        <button
                          onClick={reset}
                          className="btn-ghost flex items-center gap-2"
                        >
                          <RefreshCw className="w-4 h-4" aria-hidden="true" />
                          {t('result.newImage')}
                        </button>
                      </div>
                    </div>
                  </div>
                </motion.div>
              )}

              {state.status === 'error' && (
                <motion.div
                  key="error"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  className="max-w-xl mx-auto"
                  role="alert"
                >
                  <div className="card p-8 border-coral-100">
                    <div className="text-center">
                      <div className="w-12 h-12 mx-auto mb-4 rounded-full bg-coral-50 flex items-center justify-center">
                        <span className="text-xl text-coral" aria-hidden="true">×</span>
                      </div>
                      <h3 className="font-medium text-stone-800 mb-2">{t('errors.generic')}</h3>
                      <p className="text-sm text-stone-400 mb-6">
                        {state.error?.slice(0, 200) || t('errors.unexpected')}
                      </p>
                      <button
                        onClick={reset}
                        className="btn-prism"
                      >
                        {t('common.tryAgain')}
                      </button>
                    </div>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </motion.div>
        </div>
      </section>

      {/* How it works */}
      <section
        id="how"
        className="relative px-6 md:px-12 py-24"
        aria-labelledby="how-heading"
        style={{
          background: 'linear-gradient(180deg, transparent 0%, rgba(255, 247, 243, 0.5) 50%, transparent 100%)',
        }}
      >
        <div className="max-w-6xl mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="max-w-xl mb-16"
          >
            <h2 id="how-heading" className="text-3xl font-semibold text-stone-900 mb-4">
              {t('home.howItWorks.title')}
            </h2>
            <p className="text-stone-500">
              {t('home.howItWorks.subtitle')}
            </p>
          </motion.div>

          <div className="grid md:grid-cols-3 gap-8 md:gap-12" role="list">
            {steps.map((step, i) => (
              <StepCard
                key={step.num}
                num={step.num}
                title={step.title}
                desc={step.desc}
                delay={i * 0.1}
              />
            ))}
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="relative px-6 md:px-12 py-12 border-t border-stone-200" role="contentinfo">
        <div className="max-w-6xl mx-auto flex flex-col md:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-2 text-sm text-stone-500">
            <span className="font-medium text-stone-900">ReBloom</span>
            <span aria-hidden="true">·</span>
            <span>{t('footer.poweredBy')}</span>
          </div>

          <div className="flex items-center gap-6 text-sm text-stone-500">
            <a
              href="https://github.com/your-repo/rebloom"
              target="_blank"
              rel="noopener noreferrer"
              className="hover:text-stone-900 transition-colors"
            >
              {t('nav.github')}
            </a>
            <span className="text-stone-300" aria-hidden="true">·</span>
            <span>{t('footer.madeWith')}</span>
          </div>
        </div>
      </footer>
    </main>
  );
}
