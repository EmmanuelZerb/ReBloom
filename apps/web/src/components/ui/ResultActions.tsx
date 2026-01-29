'use client';

import { useState } from 'react';
import { motion } from 'framer-motion';
import { Download, RefreshCw, Share2, Check } from 'lucide-react';
import { cn } from '@/lib/utils';

interface ResultActionsProps {
  downloadUrl: string;
  filename: string;
  onReset: () => void;
  className?: string;
}

export function ResultActions({ downloadUrl, filename, onReset, className }: ResultActionsProps) {
  const [copied, setCopied] = useState(false);

  const handleShare = async () => {
    try {
      if (navigator.share) {
        await navigator.share({
          title: 'ReBloom - Enhanced Image',
          text: 'Check out my enhanced image!',
          url: window.location.href,
        });
      } else {
        await navigator.clipboard.writeText(window.location.href);
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
      }
    } catch (err) {
      // User cancelled share
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.1 }}
      className={cn('flex flex-wrap items-center gap-3', className)}
    >
      {/* Download Button - Primary */}
      <a
        href={downloadUrl}
        download={filename}
        className="btn-prism flex-1 flex items-center justify-center gap-2"
      >
        <Download className="w-4 h-4" />
        <span>Download</span>
      </a>

      {/* Share Button */}
      <button
        onClick={handleShare}
        className={cn(
          'btn-ghost flex items-center gap-2',
          copied && 'border-sage text-sage-dark'
        )}
      >
        {copied ? (
          <>
            <Check className="w-4 h-4" />
            <span>Copied!</span>
          </>
        ) : (
          <>
            <Share2 className="w-4 h-4" />
            <span>Share</span>
          </>
        )}
      </button>

      {/* Reset Button */}
      <button
        onClick={onReset}
        className="btn-ghost flex items-center gap-2"
      >
        <RefreshCw className="w-4 h-4" />
        <span>New</span>
      </button>
    </motion.div>
  );
}
