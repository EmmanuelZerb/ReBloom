'use client';

import { ReactCompareSlider, ReactCompareSliderImage } from 'react-compare-slider';
import { motion } from 'framer-motion';
import { ArrowLeftRight } from 'lucide-react';
import { cn } from '@/lib/utils';

interface ImageCompareProps {
  originalUrl: string;
  enhancedUrl: string;
  className?: string;
}

export function ImageCompare({ originalUrl, enhancedUrl, className }: ImageCompareProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      className={cn('relative max-w-4xl mx-auto', className)}
    >
      <div className="card p-4 overflow-hidden">
        <ReactCompareSlider
          itemOne={
            <ReactCompareSliderImage
              src={originalUrl}
              alt="Original"
              style={{ objectFit: 'contain', background: '#fafaf9' }}
            />
          }
          itemTwo={
            <ReactCompareSliderImage
              src={enhancedUrl}
              alt="Enhanced"
              style={{ objectFit: 'contain', background: '#fafaf9' }}
            />
          }
          handle={<SliderHandle />}
          className="aspect-video rounded-xl overflow-hidden"
        />

        {/* Labels */}
        <div className="absolute bottom-8 left-8 px-3 py-1.5 rounded-full bg-white shadow-soft text-xs text-stone-500">
          Original
        </div>
        <div className="absolute bottom-8 right-8 px-3 py-1.5 rounded-full bg-stone-900 text-xs text-white">
          Enhanced
        </div>
      </div>
    </motion.div>
  );
}

function SliderHandle() {
  return (
    <div className="flex items-center justify-center w-full h-full">
      <div className="relative flex items-center justify-center">
        {/* Vertical line */}
        <div className="absolute h-full w-0.5 bg-white shadow-md" />

        {/* Handle button */}
        <motion.div
          className="relative flex items-center justify-center w-10 h-10 rounded-full 
                     bg-white shadow-medium cursor-ew-resize"
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
        >
          <ArrowLeftRight className="w-4 h-4 text-stone-600" />
        </motion.div>
      </div>
    </div>
  );
}
