'use client';

import { useRouter } from '@/i18n/navigation';
import { SignupForm } from '@/components/auth/SignupForm';

export default function SignupPage() {
  const router = useRouter();

  const handleSuccess = () => {
    // Stay on page to show confirmation message
  };

  const handleSwitchToLogin = () => {
    router.push('/auth/login');
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-[#FFF8F4] via-white to-[#F3F8F5] px-4">
      <SignupForm onSuccess={handleSuccess} onSwitchToLogin={handleSwitchToLogin} />
    </div>
  );
}
