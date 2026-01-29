'use client';

import { motion, AnimatePresence } from 'framer-motion';
import { ArrowRight, Check, Download, RefreshCw, ImageIcon, User } from 'lucide-react';
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
    <main className="min-h-screen bg-white">
      {/* Header */}
      <header className="px-6 py-6 md:px-12">
        <nav className="max-w-6xl mx-auto flex items-center justify-between">
          <motion.a 
            href="/"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="text-lg font-semibold text-stone-900 tracking-tight"
          >
            ReBloom
          </motion.a>
          
          <motion.div 
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.1 }}
            className="flex items-center gap-6"
          >
            <a href="#how" className="text-sm text-stone-500 hover:text-stone-900 transition-colors">
              How it works
            </a>
            <a 
              href="https://github.com" 
              className="text-sm text-stone-500 hover:text-stone-900 transition-colors"
            >
              GitHub
            </a>
            
            {!isLoading && (
              user ? (
                <div className="flex items-center gap-4">
                  <span className="text-sm text-stone-500">{user.email?.split('@')[0]}</span>
                  <button 
                    onClick={() => signOut()}
                    className="text-sm text-stone-400 hover:text-stone-900 transition-colors"
                  >
                    Sign out
                  </button>
                </div>
              ) : (
                <Link 
                  href="/auth/login"
                  className="text-sm font-medium text-white bg-stone-900 hover:bg-stone-800 px-4 py-2 rounded-full transition-colors flex items-center gap-2"
                >
                  <User className="w-3.5 h-3.5" />
                  Sign in
                </Link>
              )
            )}
          </motion.div>
        </nav>
      </header>

      {/* Hero */}
      <section className="px-6 md:px-12 pt-12 md:pt-24 pb-20">
        <div className="max-w-6xl mx-auto">
          <div className="max-w-2xl">
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-[#FFF5F3] text-[#FF7F66] text-sm font-medium mb-4"
            >
              <span className="w-1.5 h-1.5 rounded-full bg-[#FF7F66] animate-pulse" />
              AI Image Enhancement
            </motion.div>
            
            <motion.h1
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.05 }}
              className="text-4xl md:text-6xl font-semibold text-stone-900 leading-[1.1] tracking-tight mb-6"
            >
              Make blurry photos sharp again
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
              className="flex items-center gap-5 text-sm text-stone-400"
            >
              <span className="flex items-center gap-2">
                <span className="w-5 h-5 rounded-full bg-[#F3F8F5] flex items-center justify-center">
                  <Check className="w-3 h-3 text-[#86B09B]" />
                </span>
                Free to use
              </span>
              <span className="flex items-center gap-2">
                <span className="w-5 h-5 rounded-full bg-[#F3F8F5] flex items-center justify-center">
                  <Check className="w-3 h-3 text-[#86B09B]" />
                </span>
                4x upscaling
              </span>
              <span className="flex items-center gap-2">
                <span className="w-5 h-5 rounded-full bg-[#F3F8F5] flex items-center justify-center">
                  <Check className="w-3 h-3 text-[#86B09B]" />
                </span>
                Private
              </span>
            </motion.div>
          </div>
        </div>
      </section>

      {/* Main App Area */}
      <section className="px-6 md:px-12 pb-32">
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
                >
                  <div className="w-12 h-12 mx-auto mb-6 rounded-full border-2 border-stone-200 border-t-stone-900 animate-spin" />
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
                >
                  {previewUrl && (
                    <div className="card p-4">
                      <div className="relative aspect-video rounded-xl overflow-hidden bg-stone-100">
                        <img
                          src={previewUrl}
                          alt="Original"
                          className="w-full h-full object-contain opacity-60"
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
                        <div className="w-10 h-10 rounded-full bg-[#F3F8F5] flex items-center justify-center">
                          <Check className="w-5 h-5 text-[#86B09B]" />
                        </div>
                      </div>

                      <div className="flex items-center gap-3">
                        <a
                          href={state.downloadUrl}
                          download={state.job.processedImage?.originalName || 'enhanced.png'}
                          className="btn-prism flex-1 flex items-center justify-center gap-2"
                        >
                          <Download className="w-4 h-4" />
                          Download
                        </a>
                        <button
                          onClick={reset}
                          className="btn-ghost flex items-center gap-2"
                        >
                          <RefreshCw className="w-4 h-4" />
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
                >
                  <div className="card p-8 border-[#FFE8E3]">
                    <div className="text-center">
                      <div className="w-12 h-12 mx-auto mb-4 rounded-full bg-[#FFF5F3] flex items-center justify-center">
                        <span className="text-xl text-[#FF7F66]">×</span>
                      </div>
                      <h3 className="font-medium text-stone-800 mb-2">Something went wrong</h3>
                      <p className="text-sm text-stone-400 mb-6">{state.error}</p>
                      <button onClick={reset} className="btn-prism">
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
      <section id="how" className="px-6 md:px-12 py-24 bg-gradient-to-b from-transparent via-[#FFF8F4]/50 to-transparent">
        <div className="max-w-6xl mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="max-w-xl mb-16"
          >
            <h2 className="text-3xl font-semibold text-stone-900 mb-4">
              How it works
            </h2>
            <p className="text-stone-500">
              Three simple steps. No account needed.
            </p>
          </motion.div>

          <div className="grid md:grid-cols-3 gap-8 md:gap-12">
            {[
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
            ].map((step, i) => (
              <motion.div
                key={step.num}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: i * 0.1 }}
                className="relative"
              >
                <span className="inline-flex items-center justify-center w-8 h-8 rounded-full bg-[#FFF5F3] text-[#FF7F66] text-sm font-mono mb-4">
                  {step.num}
                </span>
                <h3 className="text-xl font-medium text-stone-800 mb-2">
                  {step.title}
                </h3>
                <p className="text-stone-400 leading-relaxed">
                  {step.desc}
                </p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="px-6 md:px-12 py-12 border-t border-stone-200">
        <div className="max-w-6xl mx-auto flex flex-col md:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-2 text-sm text-stone-500">
            <span className="font-medium text-stone-900">ReBloom</span>
            <span>·</span>
            <span>Powered by Real-ESRGAN</span>
          </div>
          
          <div className="flex items-center gap-6 text-sm text-stone-500">
            <a href="https://github.com" className="hover:text-stone-900 transition-colors">
              GitHub
            </a>
            <span className="text-stone-300">·</span>
            <span>Made with care</span>
          </div>
        </div>
      </footer>
    </main>
  );
}
