import { InterviewSetupForm } from "../../../components/forms/InterviewSetupForm";

export default function NewInterviewPage() {
  return (
    <div className="min-h-screen bg-gradient-to-b from-gray-50 to-white">
      <div className="container mx-auto py-12 px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-12">
          <h1 className="text-4xl font-bold text-gray-900 mb-4">
            Set Up Your Mock Interview
          </h1>
          <p className="text-lg text-gray-600 max-w-2xl mx-auto">
            Prepare for your next job interview with our AI-powered mock
            interview system. Fill in the details below to get started.
          </p>
        </div>

        <div className="bg-white rounded-2xl shadow-xl p-6 sm:p-10 max-w-3xl mx-auto">
          <div className="grid grid-cols-1 gap-8">
            <div className="flex items-center space-x-4 pb-6 border-b border-gray-100">
              <div className="flex-shrink-0">
                <div className="bg-blue-100 p-3 rounded-full">
                  <svg
                    className="w-6 h-6 text-blue-600"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M19 21V5a2 2 0 00-2-2H7a2 2 0 00-2 2v16m14 0h2m-2 0h-5m-9 0H3m2 0h5M9 7h1m-1 4h1m4-4h1m-1 4h1m-5 10v-5a1 1 0 011-1h2a1 1 0 011 1v5m-4 0h4"
                    />
                  </svg>
                </div>
              </div>
              <div>
                <h2 className="text-xl font-semibold text-gray-900">
                  Interview Details
                </h2>
                <p className="text-gray-500 text-sm">
                  Tell us about the position you're preparing for
                </p>
              </div>
            </div>

            <InterviewSetupForm />
          </div>
        </div>
      </div>
    </div>
  );
}
