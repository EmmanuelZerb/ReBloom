'use client';

import { memo, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Check, Download, RefreshCw, User, Sparkles } from 'lucide-react';
import Link from 'next/link';
import { DropZone } from '@/components/upload/DropZone';
import { ProcessingStatus } from '@/components/progress/ProcessingStatus';
import { ImageCompare } from '@/components/compare/ImageCompare';
import { useUpload } from '@/hooks/useUpload';
import { useAuth } from '@/contexts/AuthContext';

// Memoized feature badge component
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

// Memoized step card component
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

  // Memoize steps data
  const steps = useMemo(() => [
    { 
      num: '01', 
      title: 'Upload', 
      desc: 'Drop your image or click to browse. We accept PNG, JPG, and WebP.' 
    },
    { 
      num: '02', 
      title: 'Process', 
      desc: 'Our AI analyzes and enhances every detail. Takes about 10-15 seconds.' 
    },
    { 
      num: '03', 
      title: 'Download', 
      desc: 'Get your enhanced image at 4x resolution. No watermarks, no catches.' 
    },
  ], []);

  return (
    <main className="min-h-screen bg-white relative overflow-hidden">
      {/* Skip to main content - Accessibility */}
      <a 
        href="#main-content" 
        className="sr-only focus:not-sr-only focus:absolute focus:top-4 focus:left-4 focus:z-50 focus:px-4 focus:py-2 focus:bg-stone-900 focus:text-white focus:rounded-lg"
      >
        Skip to main content
      </a>

      {/* Subtle background gradient for depth */}
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
              className="text-sm text-stone-500 hover:text-stone-900 transition-colors"
              aria-label="Learn how ReBloom works"
            >
              How it works
            </a>
            <a 
              href="https://github.com/your-repo/rebloom"
              target="_blank"
              rel="noopener noreferrer"
              className="text-sm text-stone-500 hover:text-stone-900 transition-colors"
              aria-label="View ReBloom on GitHub (opens in new tab)"
            >
              GitHub
            </a>
            
            {!isLoading && (
              user ? (
                <div className="flex items-center gap-4">
                  <span className="text-sm text-stone-500" aria-label={`Logged in as ${user.email}`}>
                    {user.email?.split('@')[0]}
                  </span>
                  <button 
                    onClick={() => signOut()}
                    className="text-sm text-stone-400 hover:text-stone-900 transition-colors"
                    aria-label="Sign out of your account"
                  >
                    Sign out
                  </button>
                </div>
              ) : (
                <Link 
                  href="/auth/login"
                  className="text-sm font-medium text-white bg-stone-900 hover:bg-stone-800 px-4 py-2 rounded-full transition-colors flex items-center gap-2"
                  aria-label="Sign in to your account"
                >
                  <User className="w-3.5 h-3.5" aria-hidden="true" />
                  Sign in
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
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-coral-50 text-coral text-sm font-medium mb-4"
            >
              <span className="w-1.5 h-1.5 rounded-full bg-coral animate-pulse" aria-hidden="true" />
              AI Image Enhancement
            </motion.div>
            
            <motion.h1
              id="hero-heading"
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.05 }}
              className="text-4xl md:text-6xl font-semibold text-stone-900 leading-[1.1] tracking-tight mb-6"
            >
              Make blurry photos{' '}
              <span className="relative">
                sharp again
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
              Upload any photo and watch it transform. 4x upscaling, detail restoration, 
              no watermarks. Your images stay private.
            </motion.p>

            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.15 }}
              className="flex flex-wrap items-center gap-5 text-sm text-stone-400"
              role="list"
              aria-label="Key features"
            >
              <FeatureBadge>Free to use</FeatureBadge>
              <FeatureBadge>4x upscaling</FeatureBadge>
              <FeatureBadge>Private</FeatureBadge>
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
              {/* Idle State - Upload */}
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

              {/* Uploading State */}
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
                  <p className="text-stone-500">Uploading your image...</p>
                </motion.div>
              )}

              {/* Processing State */}
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
                            Processing...
                          </span>
                        </div>
                      </div>
                    </div>
                  )}
                  <ProcessingStatus job={state.job} />
                </motion.div>
              )}

              {/* Completed State */}
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
                          <h3 className="font-medium text-stone-900">Ready to download</h3>
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
                          aria-label={`Download enhanced image ${state.job.processedImage?.originalName || 'enhanced.png'}`}
                        >
                          <Download className="w-4 h-4" aria-hidden="true" />
                          Download
                        </a>
                        <button
                          onClick={reset}
                          className="btn-ghost flex items-center gap-2"
                          aria-label="Start over with a new image"
                        >
                          <RefreshCw className="w-4 h-4" aria-hidden="true" />
                          New image
                        </button>
                      </div>
                    </div>
                  </div>
                </motion.div>
              )}

              {/* Error State */}
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
                      <h3 className="font-medium text-stone-800 mb-2">Something went wrong</h3>
                      <p className="text-sm text-stone-400 mb-6">
                        {/* Sanitize error message - only show safe content */}
                        {state.error?.slice(0, 200) || 'An unexpected error occurred'}
                      </p>
                      <button 
                        onClick={reset} 
                        className="btn-prism"
                        aria-label="Try uploading again"
                      >
                        Try again
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
              How it works
            </h2>
            <p className="text-stone-500">
              Three simple steps. No account needed.
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
            <span>Powered by Real-ESRGAN</span>
          </div>
          
          <div className="flex items-center gap-6 text-sm text-stone-500">
            <a 
              href="https://github.com/your-repo/rebloom"
              target="_blank"
              rel="noopener noreferrer"
              className="hover:text-stone-900 transition-colors"
              aria-label="View ReBloom on GitHub (opens in new tab)"
            >
              GitHub
            </a>
            <span className="text-stone-300" aria-hidden="true">·</span>
            <span>Made with care</span>
          </div>
        </div>
      </footer>
    </main>
  );
}
