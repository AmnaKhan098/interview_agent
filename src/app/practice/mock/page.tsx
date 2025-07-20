"use client";
import { Button } from "../../../components/ui/Button";
import { RxArrowRight } from "react-icons/rx";
import { useRouter } from "next/navigation";

export default function MockInterviewPage() {
  const router = useRouter();
  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-blue-50 via-white to-blue-100 py-20 px-4 animate-fade-in">
      <div className="bg-white/90 shadow-2xl rounded-3xl p-12 max-w-lg w-full text-center border border-blue-100 backdrop-blur-md">
        <h1 className="text-4xl font-extrabold text-blue-900 mb-5 tracking-tight drop-shadow-sm">
          Mock Interview
        </h1>
        <p className="text-gray-700 mb-10 text-lg leading-relaxed">
          Practice real interview questions in a timed, realistic environment.
          <br />
          Sharpen your skills and boost your confidence before the real thing.
        </p>
        <Button
          size="lg"
          className="w-full flex items-center justify-center gap-2 bg-blue-600 hover:bg-blue-700 text-white font-semibold py-4 rounded-xl shadow-lg transition-all duration-200 text-lg group"
          onClick={() => router.push("/interview/new")}
        >
          Start Mock Interview
          <RxArrowRight className="ml-2 w-5 h-5 group-hover:translate-x-1 transition-transform duration-200" />
        </Button>
      </div>
      <style jsx global>{`
        .animate-fade-in {
          animation: fadeIn 0.8s cubic-bezier(0.4, 0, 0.2, 1);
        }
        @keyframes fadeIn {
          from {
            opacity: 0;
            transform: translateY(30px);
          }
          to {
            opacity: 1;
            transform: none;
          }
        }
      `}</style>
    </div>
  );
}
