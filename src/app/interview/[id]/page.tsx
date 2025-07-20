"use client";

import { useState, useEffect, useRef } from "react";
import { useRouter, useSearchParams, useParams } from "next/navigation";


// Mock interview questions


type InterviewState = "idle" | "asking" | "listening" | "finished";

const QUESTION_TIME = 60; // seconds per question
const INTERVIEW_TIME = 5 * 60; // total interview time (5 min for 5 questions)

export default function InterviewPage() {
  const router = useRouter();
  const params = useParams();
  const interviewId = params.id as string;
  const videoRef = useRef<HTMLVideoElement>(null);
  const [currentQuestionIndex, setCurrentQuestionIndex] = useState(0);
  const [interviewState, setInterviewState] = useState<InterviewState>("idle");
  const [transcript, setTranscript] = useState("");
  const [error, setError] = useState("");
  const [questionTimer, setQuestionTimer] = useState(QUESTION_TIME);
  const [interviewTimer, setInterviewTimer] = useState(INTERVIEW_TIME);
  const [questions, setQuestions] = useState<string[]>([]);
  const [answers, setAnswers] = useState<string[]>([]);
  const [jobTitle, setJobTitle] = useState<string>("");
  const [loadingFeedback, setLoadingFeedback] = useState(false);
  const recognitionRef = useRef<any>(null);

  useEffect(() => {
    const fetchInterviewDetails = async () => {
      const urlParams = new URLSearchParams(window.location.search);
      const questionsParam = urlParams.get('questions');
      if (questionsParam) {
        try {
          const parsedQuestions = JSON.parse(decodeURIComponent(questionsParam));
          setQuestions(parsedQuestions);
          setAnswers(Array(parsedQuestions.length).fill(""));
        } catch (e) {
          console.error("Failed to parse questions from URL", e);
        }
      }

      // Fetch job title from API
      try {
        const response = await fetch(`/api/interview/${interviewId}`);
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        if (data.jobTitle) {
          setJobTitle(data.jobTitle);
        }
      } catch (e) {
        console.error("Failed to fetch interview details:", e);
      }
    };

    fetchInterviewDetails();
  }, [interviewId]);

  // Initialize webcam
  useEffect(() => {
    async function setupWebcam() {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: true,
          audio: true,
        });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
      } catch (err) {
        setError("Camera access denied. Please check your permissions.");
      }
    }
    setupWebcam();
  }, []);

  // Interview and question timers
  useEffect(() => {
    let interviewInterval: NodeJS.Timeout | null = null;
    let questionInterval: NodeJS.Timeout | null = null;
    if (interviewState === "asking" || interviewState === "listening") {
      interviewInterval = setInterval(() => {
        setInterviewTimer((t) => {
          if (t <= 1) {
            clearInterval(interviewInterval!); 
            setInterviewState("finished");
            processFeedback();
            return 0;
          }
          return t - 1;
        });
      }, 1000);
      questionInterval = setInterval(() => {
        setQuestionTimer((t) => {
          if (t <= 1) {
            clearInterval(questionInterval!);
            stopListening();
            goToNextQuestion();
            return QUESTION_TIME;
          }
          return t - 1;
        });
      }, 1000);
    }
    return () => {
      if (interviewInterval) clearInterval(interviewInterval);
      if (questionInterval) clearInterval(questionInterval);
    };
  }, [interviewState, currentQuestionIndex, questions, answers, router, interviewId]);

  // Start question
  useEffect(() => {
    if (interviewState === "asking") {
      setQuestionTimer(QUESTION_TIME);
      setTranscript("");
      startListening();
    }
  }, [interviewState, currentQuestionIndex]);

  // Add TTS effect for each question
  useEffect(() => {
    if (interviewState === "asking" && window.speechSynthesis && questions.length > 0) {
      const utterance = new window.SpeechSynthesisUtterance(
        questions[currentQuestionIndex]
      );
      window.speechSynthesis.cancel(); // Stop any previous speech
      window.speechSynthesis.speak(utterance);
      utterance.onend = () => {
        setInterviewState("listening");
      };
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [interviewState, currentQuestionIndex, questions]);

  // Navigation
  const goToNextQuestion = () => {
    // Save the current answer before moving to the next question
    setAnswers((prev) => {
      const updated = [...prev];
      updated[currentQuestionIndex] = transcript.trim(); // Save current transcript as the answer, or empty if skipped
      return updated;
    });

    if (currentQuestionIndex < questions.length - 1) {
      setCurrentQuestionIndex((prev) => prev + 1);
      setInterviewState("asking");
    } else {
      setInterviewState("finished");
      processFeedback();
    }
  };
  const goToPrevQuestion = () => {
    if (currentQuestionIndex > 0) {
      setCurrentQuestionIndex((prev) => prev - 1);
      setInterviewState("asking");
    }
  };

  // Speech recognition
  const startListening = () => {
    if (!("webkitSpeechRecognition" in window)) {
      setError("Speech recognition is not supported in your browser.");
      return;
    }
    const SpeechRecognition = (window as any).webkitSpeechRecognition;
    const recognition = new SpeechRecognition();
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.onresult = (event: any) => {
      const transcript = Array.from(event.results)
        .map((result: unknown) => (result as SpeechRecognitionResult)[0])
        .map((result) => result.transcript)
        .join("");
      setTranscript(transcript);
      setAnswers((prev) => {
        const updated = [...prev];
        updated[currentQuestionIndex] = transcript;
        return updated;
      });
    };
    recognition.onend = () => {};
    recognitionRef.current = recognition;
    recognition.start();
    setInterviewState("listening");
  };
  const stopListening = () => {
    if (recognitionRef.current) {
      recognitionRef.current.stop();
      recognitionRef.current = null;
    }
  };

  const processFeedback = async () => {
    setLoadingFeedback(true);
    try {
      const response = await fetch('/api/process-feedback', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          interviewId,
          questions,
          answers,
          jobTitle,
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      console.log('Received feedback data:', data);
      if (data) {
        sessionStorage.setItem('interviewFeedback', JSON.stringify(data));
        router.push(`/feedback?interviewId=${interviewId}`);
      } else {
        setError('Received empty or invalid feedback data.');
        console.error('Received empty or invalid feedback data:', data);
      }
    } catch (error) {
      console.error('Error processing feedback:', error);
      setError('Failed to process feedback.');
    } finally {
      setLoadingFeedback(false);
    }
  };

  // Add finish handler
  const finishInterview = async () => {
    // Ensure all questions have been answered before finishing
    if (currentQuestionIndex < questions.length - 1) {
      alert("Please complete all questions before ending the interview.");
      return;
    }

    // Save the last answer
    setAnswers((prev) => {
      const updated = [...prev];
      updated[currentQuestionIndex] = transcript.trim();
      return updated;
    });

    // Proceed to feedback page
    processFeedback();

     // Proceed to feedback page
     setInterviewState("finished");
     processFeedback();

  };


  return (
    <div className="min-h-screen bg-gray-50 py-8 px-4 sm:px-6 lg:px-8">
      <div className="max-w-7xl mx-auto">
        {/* Interview Progress */}
        <div className="mb-8">
          <div className="flex items-center justify-between mb-4">
            <h1 className="text-2xl font-bold text-gray-900">
              Technical Interview Session
            </h1>
            <span className="text-sm font-medium text-gray-500">
              Question {currentQuestionIndex + 1} of {questions.length}
            </span>
          </div>
          <div className="w-full bg-gray-200 rounded-full h-2">
            <div
              className="bg-blue-600 h-2 rounded-full transition-all duration-300"
              style={{
                width: `${
                  ((currentQuestionIndex + 1) / questions.length) * 100
                }%`,
              }}
            />
          </div>
          <div className="flex justify-between mt-2 text-sm text-gray-600">
            <span>
              Interview Time Left: {Math.floor(interviewTimer / 60)}:
              {String(interviewTimer % 60).padStart(2, "0")}
            </span>
            <span>Question Time Left: {questionTimer}s</span>
          </div>
        </div>
        <div className="flex justify-end mb-4">
          <button
            onClick={finishInterview}
            disabled={questions.length === 0}
            className="px-4 py-2 bg-red-600 text-white rounded-lg font-semibold shadow hover:bg-red-700 transition-colors"
          >
            Finish Interview
          </button>
        </div>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Left Column - AI Agent */}
          <div className="space-y-6">
            <div className="bg-gradient-to-br from-blue-500 to-purple-600 w-full aspect-video rounded-2xl shadow-lg flex items-center justify-center">
              <div className="text-center">
                <span className="text-white text-xl font-medium">
                  AI Interviewer
                </span>
                <div className="mt-2">
                  {interviewState === "asking" && (
                    <div className="flex items-center justify-center space-x-2">
                      <div
                        className="w-2 h-2 bg-white rounded-full animate-bounce"
                        style={{ animationDelay: "0s" }}
                      />
                      <div
                        className="w-2 h-2 bg-white rounded-full animate-bounce"
                        style={{ animationDelay: "0.2s" }}
                      />
                      <div
                        className="w-2 h-2 bg-white rounded-full animate-bounce"
                        style={{ animationDelay: "0.4s" }}
                      />
                    </div>
                  )}
                </div>
              </div>
            </div>
            <div className="bg-white p-8 rounded-2xl shadow-lg">
              <h2 className="text-xl font-semibold text-gray-900 mb-4">
                Current Question:
              </h2>
              <p className="text-lg text-gray-700 leading-relaxed">
                {questions[currentQuestionIndex]}
              </p>
              <div className="flex gap-2 mt-4">
                <button
                  onClick={goToPrevQuestion}
                  disabled={currentQuestionIndex === 0}
                  className={`px-4 py-2 rounded-lg font-medium transition-all duration-200 ${
                    currentQuestionIndex === 0
                      ? "bg-gray-200 text-gray-500 cursor-not-allowed"
                      : "bg-blue-100 text-blue-700 hover:bg-blue-200"
                  }`}
                >
                  Previous
                </button>
                <button
                  onClick={goToNextQuestion}
                  disabled={currentQuestionIndex === questions.length - 1}
                  className={`px-4 py-2 rounded-lg font-medium transition-all duration-200 ${
                    currentQuestionIndex === questions.length - 1
                      ? "bg-gray-200 text-gray-500 cursor-not-allowed"
                      : "bg-blue-100 text-blue-700 hover:bg-blue-200"
                  }`}
                >
                  Next
                </button>
              </div>
            </div>
          </div>
          {/* Right Column - User */}
          <div className="space-y-6">
            {error ? (
              <div className="bg-red-50 border-l-4 border-red-500 p-6 rounded-lg">
                <div className="flex items-center">
                  <svg
                    className="h-6 w-6 text-red-500 mr-3"
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
                    />
                  </svg>
                  <p className="text-red-700">{error}</p>
                </div>
              </div>
            ) : (
              <>
                <div className="relative">
                  <video
                    ref={videoRef}
                    autoPlay
                    playsInline
                    muted
                    className="w-full aspect-video rounded-2xl bg-black shadow-lg"
                  />
                  {interviewState === "listening" && (
                    <div className="absolute top-4 right-4 flex items-center space-x-2 bg-black bg-opacity-50 rounded-full px-4 py-2">
                      <div className="w-3 h-3 rounded-full bg-red-500 animate-pulse" />
                      <span className="text-white text-sm font-medium">
                        Recording
                      </span>
                    </div>
                  )}
                </div>
                <div className="flex items-center gap-4">
                  <button
                    onClick={() =>
                      interviewState === "idle" && setInterviewState("asking")
                    }
                    disabled={interviewState !== "idle"}
                    className={`px-6 py-3 rounded-lg font-medium transition-all duration-200 ${
                      interviewState === "idle"
                        ? "bg-blue-600 text-white hover:bg-blue-700"
                        : "bg-gray-200 text-gray-500 cursor-not-allowed"
                    }`}
                  >
                    Start Interview
                  </button>
                </div>
                {answers[currentQuestionIndex] && (
                  <div className="bg-white p-6 rounded-2xl shadow-lg border border-gray-100">
                    <h3 className="text-lg font-semibold text-gray-900 mb-3">
                      Your Answer:
                    </h3>
                    <p className="text-gray-700 leading-relaxed">
                      {answers[currentQuestionIndex]}
                    </p>
                  </div>
                )}
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
