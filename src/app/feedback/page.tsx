"use client";
import React, { useEffect, useState } from "react";
import { useRouter } from "next/navigation";

interface QuestionFeedbackItem {
  question: string;
  user_answer: string;
  feedback: string;
  score: number;
}

interface FeedbackData {
  overallFeedback: string | null;
  overallScore: number | null;
  feedbackResults: QuestionFeedbackItem[];
}

export default function FeedbackPage() {
  const router = useRouter();
  const [feedbackData, setFeedbackData] = useState<FeedbackData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const loadFeedback = () => {
      console.log('Attempting to load feedback...');
      const storedFeedback = sessionStorage.getItem('interviewFeedback');
      if (storedFeedback) {
        console.log('Found stored feedback in sessionStorage.');
        try {
          const parsedData = JSON.parse(storedFeedback);
          setFeedbackData(parsedData);
          console.log('Successfully parsed and set feedback data:', parsedData);
        } catch (e) {
          console.error("Error parsing stored feedback:", e);
          setError("Failed to load feedback data. Check console for details.");
        }
      } else {
        console.log('No feedback data found in sessionStorage.');
        setError("No feedback data found. Please complete an interview first.");
      }
      setLoading(false); // Always set loading to false after attempting to load
      console.log('Loading state set to false.');
    };

    if (router.isReady) {
      console.log('Router is ready. Calling loadFeedback().');
      loadFeedback();
    } else {
      console.log('Router is not ready yet. Setting a timeout to check again.');
      // Fallback for router.isReady not becoming true
      const timeoutId = setTimeout(() => {
        console.log('Timeout triggered. Forcing loadFeedback().');
        loadFeedback();
      }, 3000); // Wait for 3 seconds

      return () => clearTimeout(timeoutId); // Cleanup timeout
    }
  }, [router.isReady]);

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-b from-white to-gray-50 flex flex-col items-center justify-center px-4">
        <p>Generating feedback...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-gradient-to-b from-white to-gray-50 flex flex-col items-center justify-center px-4">
        <p className="text-red-500">Error: {error}</p>
      </div>
    );
  }

  if (!feedbackData) {
    return (
      <div className="min-h-screen bg-gradient-to-b from-white to-gray-50 flex flex-col items-center justify-center px-4">
        <p>No feedback data available.</p>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-b from-white to-gray-50 flex flex-col items-center justify-center px-4">
      <div className="bg-white rounded-2xl shadow-xl p-8 max-w-2xl w-full">
        <h1 className="text-3xl font-bold text-center text-blue-700 mb-6">
          Interview Feedback
        </h1>
        {feedbackData.overallFeedback && (
          <div className="bg-blue-50 p-6 rounded-2xl shadow-lg border border-blue-100 mb-6">
            <h3 className="text-xl font-semibold text-blue-800 mb-3">Overall Summary:</h3>
            <p className="text-blue-700 leading-relaxed mb-4">{feedbackData.overallFeedback}</p>
            {feedbackData.overallScore !== null && (
              <p className="text-lg font-bold text-blue-600">
                Overall Score: {feedbackData.overallScore.toFixed(2)}/10
              </p>
            )}
          </div>
        )}
        <div className="space-y-6">
          {feedbackData.feedbackResults.map((item, idx) => (
            <div key={idx} className="bg-white p-6 rounded-2xl shadow-lg border border-gray-100">
              <h3 className="text-xl font-semibold text-gray-900 mb-3">
                Question {idx + 1}: {item.question}
              </h3>
              <p className="text-gray-700 leading-relaxed mb-2"><strong>Your Answer:</strong> {item.user_answer}</p>
              <p className="text-gray-700 leading-relaxed mb-4"><strong>Feedback:</strong> {item.feedback}</p>
              <div className="flex items-center justify-between">
                <p className="text-lg font-bold text-blue-600">
                  Score: {item.score}/10
                </p>
              </div>
            </div>
          ))}
        </div>
        <div className="flex justify-center">
          <button
            className="bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-6 rounded-lg shadow transition-colors"
            onClick={() => router.push("/dashboard")}
          >
            Go to Dashboard
          </button>
        </div>
      </div>
    </div>
  );
}
