'use client';

import { useCallback, useState, useId } from 'react';
import { useDropzone } from 'react-dropzone';
import { motion, AnimatePresence } from 'framer-motion';
import { Upload, X, ArrowRight, ImageIcon } from 'lucide-react';
import { cn, formatBytes } from '@/lib/utils';
import { UPLOAD_CONFIG } from '@rebloom/shared';

interface DropZoneProps {
  onFileSelect: (file: File) => void;
  disabled?: boolean;
  className?: string;
  'aria-label'?: string;
}

export function DropZone({ 
  onFileSelect, 
  disabled, 
  className,
  'aria-label': ariaLabel = 'Upload image for enhancement'
}: DropZoneProps) {
  const [preview, setPreview] = useState<string | null>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const dropzoneId = useId();
  const descriptionId = useId();

  const onDrop = useCallback((acceptedFiles: File[]) => {
    const file = acceptedFiles[0];
    if (file) {
      setSelectedFile(file);
      setPreview(URL.createObjectURL(file));
    }
  }, []);

  const handleSubmit = () => {
    if (selectedFile) {
      onFileSelect(selectedFile);
    }
  };

  const handleClear = () => {
    if (preview) {
      URL.revokeObjectURL(preview);
    }
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
    <div className={cn('w-full max-w-2xl mx-auto', className)}>
      <AnimatePresence mode="wait">
        {preview ? (
          <motion.div
            key="preview"
            initial={{ opacity: 0, scale: 0.98 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.98 }}
            transition={{ duration: 0.2 }}
            className="space-y-4"
          >
            {/* Preview Image */}
            <div className="card p-4 group">
              <div className="relative aspect-video rounded-xl overflow-hidden bg-stone-100">
                <img
                  src={preview}
                  alt="Preview"
                  className="w-full h-full object-contain"
                />
                
                {/* Clear button */}
                <button
                  onClick={handleClear}
                  className="absolute top-3 right-3 p-2 rounded-full bg-white shadow-soft
                           hover:bg-stone-50 transition-colors"
                  aria-label="Remove image"
                >
                  <X className="w-4 h-4 text-stone-600" />
                </button>

                {/* File info */}
                <div className="absolute bottom-3 left-3 right-3 flex items-center justify-between
                              opacity-0 group-hover:opacity-100 transition-opacity">
                  <span className="px-3 py-1.5 rounded-full bg-white shadow-soft text-xs text-stone-600 truncate max-w-[60%]">
                    {selectedFile?.name}
                  </span>
                  <span className="px-3 py-1.5 rounded-full bg-white shadow-soft text-xs font-mono text-stone-500">
                    {selectedFile && formatBytes(selectedFile.size)}
                  </span>
                </div>
              </div>
            </div>

            {/* Submit Button */}
            <motion.button
              onClick={handleSubmit}
              disabled={disabled}
              className="btn-prism w-full flex items-center justify-center gap-2 group"
              whileTap={{ scale: 0.98 }}
            >
              <span>Enhance image</span>
              <ArrowRight className="w-4 h-4 transition-transform group-hover:translate-x-0.5" />
            </motion.button>
          </motion.div>
        ) : (
          <motion.div
            key="dropzone"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            {...getRootProps()}
            role="button"
            aria-label={ariaLabel}
            aria-disabled={disabled}
            tabIndex={disabled ? -1 : 0}
            className={cn(
              'relative cursor-pointer transition-all duration-200',
              'p-12 md:p-16 flex flex-col items-center justify-center gap-4',
              'rounded-2xl border-2 border-dashed bg-white',
              'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#FF7F66] focus-visible:ring-offset-2',
              isDragActive && !isDragReject && 'border-[#FF9B85] bg-[#FFF5F3]',
              isDragReject && 'border-[#FF7F66] bg-[#FFF5F3]',
              !isDragActive && 'border-stone-200 hover:border-[#FF9B85] hover:bg-[#FFF5F3]',
              disabled && 'opacity-50 cursor-not-allowed'
            )}
          >
            <input 
              {...getInputProps()} 
              id={dropzoneId}
              aria-describedby={descriptionId}
            />
            <span id={descriptionId} className="sr-only">
              Supported formats: PNG, JPG, WebP. Maximum file size: {UPLOAD_CONFIG.maxSizeBytes / 1024 / 1024}MB
            </span>

            {/* Icon */}
            <div
              className={cn(
                'p-4 rounded-full transition-colors',
                isDragReject 
                  ? 'bg-[#FFE8E3]' 
                  : isDragActive 
                    ? 'bg-[#FFE8E3]' 
                    : 'bg-stone-100'
              )}
            >
              {isDragReject ? (
                <X className="w-6 h-6 text-[#FF7F66]" />
              ) : (
                <Upload className={cn(
                  'w-6 h-6 transition-colors',
                  isDragActive ? 'text-[#FF7F66]' : 'text-stone-400'
                )} />
              )}
            </div>

            {/* Text */}
            <div className="text-center">
              <p className={cn(
                'text-base font-medium mb-1',
                isDragReject ? 'text-[#FF7F66]' : isDragActive ? 'text-[#FF7F66]' : 'text-stone-700'
              )}>
                {isDragReject 
                  ? 'File not supported' 
                  : isDragActive 
                    ? 'Drop to upload' 
                    : 'Drop your image here'}
              </p>
              <p className="text-sm text-stone-400">
                or <span className="text-[#FF7F66] hover:text-[#E86B52] cursor-pointer">browse files</span>
              </p>
            </div>

            {/* Formats */}
            <div className="flex items-center gap-2 text-xs text-stone-400">
              <span className="px-2 py-1 rounded bg-stone-100">PNG</span>
              <span className="px-2 py-1 rounded bg-stone-100">JPG</span>
              <span className="px-2 py-1 rounded bg-stone-100">WebP</span>
              <span className="mx-1">·</span>
              <span>Max {UPLOAD_CONFIG.maxSizeBytes / 1024 / 1024}MB</span>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
