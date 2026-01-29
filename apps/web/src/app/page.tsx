'use client';

import { motion, AnimatePresence } from 'framer-motion';
import { Check, Download, RefreshCw, User } from 'lucide-react';
import Link from 'next/link';
import { DropZone } from '@/components/upload/DropZone';
import { ProcessingStatus } from '@/components/progress/ProcessingStatus';
import { ImageCompare } from '@/components/compare/ImageCompare';
import { useUpload } from '@/hooks/useUpload';
import { useAuth } from '@/contexts/AuthContext';

export default function Home() {
  const { state, upload, reset, previewUrl } = useUpload();
  const { user, signOut, isLoading } = useAuth();

  return (
    <main className="min-h-screen bg-[#FAFAF9]">
      {/* Header */}
      <header className="px-6 py-5 md:px-12 border-b border-stone-200">
        <nav className="max-w-5xl mx-auto flex items-center justify-between">
          <a href="/" className="text-lg font-semibold text-stone-900">
            ReBloom
          </a>
          
          <div className="flex items-center gap-6">
            <a 
              href="#how" 
              className="text-sm text-stone-500 hover:text-stone-900 transition-colors hidden sm:block"
            >
              How it works
            </a>
            
            {!isLoading && (
              user ? (
                <div className="flex items-center gap-4">
                  <span className="text-sm text-stone-500">{user.email?.split('@')[0]}</span>
                  <button 
                    onClick={() => signOut()}
                    className="text-sm text-stone-500 hover:text-stone-900 transition-colors"
                  >
                    Sign out
                  </button>
                </div>
              ) : (
                <Link 
                  href="/auth/login"
                  className="text-sm font-medium text-stone-900 flex items-center gap-2"
                >
                  <User className="w-4 h-4" />
                  Sign in
                </Link>
              )
            )}
          </div>
        </nav>
      </header>

      {/* Hero */}
      <section className="px-6 md:px-12 pt-16 md:pt-24 pb-12">
        <div className="max-w-5xl mx-auto">
          <div className="max-w-xl">
            <h1 className="text-3xl md:text-4xl font-semibold text-stone-900 leading-tight mb-4">
              Enhance your images with AI
            </h1>
            <p className="text-stone-500 mb-6">
              4× upscaling. Detail restoration. No watermarks. Free.
            </p>
          </div>
        </div>
      </section>

      {/* App */}
      <section className="px-6 md:px-12 pb-24">
        <div className="max-w-5xl mx-auto">
          <AnimatePresence mode="wait">
            {state.status === 'idle' && (
              <motion.div
                key="idle"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
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
                className="card p-12 text-center max-w-lg mx-auto"
              >
                <div className="w-8 h-8 mx-auto mb-4 border-2 border-stone-300 border-t-stone-900 rounded-full animate-spin" />
                <p className="text-stone-600">Uploading...</p>
              </motion.div>
            )}

            {state.status === 'processing' && (
              <motion.div
                key="processing"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="max-w-lg mx-auto space-y-4"
              >
                {previewUrl && (
                  <div className="card p-3">
                    <div className="relative rounded-xl overflow-hidden bg-stone-100">
                      <img src={previewUrl} alt="Processing" className="w-full opacity-50" />
                      <div className="absolute inset-0 flex items-center justify-center">
                        <span className="px-3 py-1.5 bg-white rounded-full text-sm text-stone-600 shadow-sm">
                          Processing...
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
                className="space-y-6"
              >
                {previewUrl && state.job.processedImage && (
                  <ImageCompare
                    originalUrl={previewUrl}
                    enhancedUrl={state.job.processedImage.url}
                  />
                )}

                <div className="max-w-lg mx-auto">
                  <div className="card p-5">
                    <div className="flex items-center justify-between mb-4">
                      <div>
                        <p className="font-medium text-stone-900">Done</p>
                        <p className="text-sm text-stone-500">
                          {state.job.processedImage?.width} × {state.job.processedImage?.height}px
                        </p>
                      </div>
                      <div className="w-8 h-8 rounded-full bg-stone-100 flex items-center justify-center">
                        <Check className="w-4 h-4 text-stone-600" />
                      </div>
                    </div>

                    <div className="flex gap-3">
                      <a
                        href={state.downloadUrl}
                        download
                        className="btn-primary flex-1 flex items-center justify-center gap-2"
                      >
                        <Download className="w-4 h-4" />
                        Download
                      </a>
                      <button onClick={reset} className="btn-secondary flex items-center gap-2">
                        <RefreshCw className="w-4 h-4" />
                        New
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
                className="max-w-lg mx-auto"
              >
                <div className="card p-8 text-center">
                  <p className="font-medium text-stone-900 mb-2">Error</p>
                  <p className="text-sm text-stone-500 mb-4">{state.error}</p>
                  <button onClick={reset} className="btn-primary">
                    Try again
                  </button>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </section>

      {/* How it works */}
      <section id="how" className="px-6 md:px-12 py-16 border-t border-stone-200">
        <div className="max-w-5xl mx-auto">
          <h2 className="text-xl font-semibold text-stone-900 mb-8">How it works</h2>
          
          <div className="grid md:grid-cols-3 gap-8">
            {[
              { num: '1', title: 'Upload', desc: 'PNG, JPG, or WebP up to 10MB' },
              { num: '2', title: 'Process', desc: 'AI enhances details in ~15s' },
              { num: '3', title: 'Download', desc: '4× resolution, no watermarks' },
            ].map((step) => (
              <div key={step.num}>
                <span className="inline-block w-6 h-6 rounded-full bg-stone-900 text-white text-sm font-medium text-center leading-6 mb-3">
                  {step.num}
                </span>
                <h3 className="font-medium text-stone-900 mb-1">{step.title}</h3>
                <p className="text-sm text-stone-500">{step.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="px-6 md:px-12 py-6 border-t border-stone-200">
        <div className="max-w-5xl mx-auto flex items-center justify-between text-sm text-stone-500">
          <span>ReBloom · Real-ESRGAN</span>
          <a 
            href="https://github.com" 
            target="_blank" 
            rel="noopener noreferrer"
            className="hover:text-stone-900 transition-colors"
          >
            GitHub
          </a>
        </div>
      </footer>
    </main>
  );
}
