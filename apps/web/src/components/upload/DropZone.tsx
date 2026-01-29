'use client';

import { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { motion, AnimatePresence } from 'framer-motion';
import { Upload, X, ArrowRight } from 'lucide-react';
import { cn, formatBytes } from '@/lib/utils';
import { UPLOAD_CONFIG } from '@rebloom/shared';

interface DropZoneProps {
  onFileSelect: (file: File) => void;
  disabled?: boolean;
  className?: string;
}

export function DropZone({ onFileSelect, disabled, className }: DropZoneProps) {
  const [preview, setPreview] = useState<string | null>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);

  const onDrop = useCallback((acceptedFiles: File[]) => {
    const file = acceptedFiles[0];
    if (file) {
      setSelectedFile(file);
      setPreview(URL.createObjectURL(file));
    }
  }, []);

  const handleSubmit = () => {
    if (selectedFile) onFileSelect(selectedFile);
  };

  const handleClear = () => {
    if (preview) URL.revokeObjectURL(preview);
    setPreview(null);
    setSelectedFile(null);
  };

  const { getRootProps, getInputProps, isDragActive, isDragReject } = useDropzone({
    onDrop,
    accept: {
      'image/jpeg': ['.jpg', '.jpeg'],
      'image/png': ['.png'],
      'image/webp': ['.webp'],
    },
    maxSize: UPLOAD_CONFIG.maxSizeBytes,
    multiple: false,
    disabled,
  });

  return (
    <div className={cn('w-full max-w-lg mx-auto', className)}>
      <AnimatePresence mode="wait">
        {preview ? (
          <motion.div
            key="preview"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="space-y-4"
          >
            <div className="card p-3">
              <div className="relative rounded-xl overflow-hidden bg-stone-100">
                <img src={preview} alt="Preview" className="w-full" />
                <button
                  onClick={handleClear}
                  className="absolute top-2 right-2 p-2 bg-white rounded-full shadow-sm hover:bg-stone-50"
                >
                  <X className="w-4 h-4 text-stone-600" />
                </button>
                <div className="absolute bottom-2 left-2 right-2 flex justify-between">
                  <span className="px-2 py-1 bg-white/90 rounded text-xs text-stone-600 truncate max-w-[60%]">
                    {selectedFile?.name}
                  </span>
                  <span className="px-2 py-1 bg-white/90 rounded text-xs text-stone-500 font-mono">
                    {selectedFile && formatBytes(selectedFile.size)}
                  </span>
                </div>
              </div>
            </div>

            <button
              onClick={handleSubmit}
              disabled={disabled}
              className="btn-primary w-full flex items-center justify-center gap-2"
            >
              Enhance
              <ArrowRight className="w-4 h-4" />
            </button>
          </motion.div>
        ) : (
          <motion.div
            key="dropzone"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            {...getRootProps()}
            className={cn(
              'p-12 flex flex-col items-center justify-center gap-4 cursor-pointer',
              'rounded-2xl border-2 border-dashed transition-colors',
              isDragActive && !isDragReject && 'border-stone-400 bg-stone-50',
              isDragReject && 'border-red-400 bg-red-50',
              !isDragActive && 'border-stone-300 hover:border-stone-400 hover:bg-stone-50',
              disabled && 'opacity-50 cursor-not-allowed'
            )}
          >
            <input {...getInputProps()} />

            <div className={cn(
              'p-4 rounded-full',
              isDragReject ? 'bg-red-100' : 'bg-stone-100'
            )}>
              {isDragReject ? (
                <X className="w-6 h-6 text-red-500" />
              ) : (
                <Upload className="w-6 h-6 text-stone-400" />
              )}
            </div>

            <div className="text-center">
              <p className={cn(
                'font-medium mb-1',
                isDragReject ? 'text-red-600' : 'text-stone-700'
              )}>
                {isDragReject ? 'Not supported' : isDragActive ? 'Drop here' : 'Drop an image'}
              </p>
              {!isDragActive && (
                <p className="text-sm text-stone-400">
                  or <span className="text-stone-600 underline">browse</span>
                </p>
              )}
            </div>

            <div className="flex items-center gap-2 text-xs text-stone-400">
              <span>PNG, JPG, WebP</span>
              <span>·</span>
              <span>Max {UPLOAD_CONFIG.maxSizeBytes / 1024 / 1024}MB</span>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
