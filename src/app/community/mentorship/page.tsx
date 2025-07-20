import { Button } from "../../../components/ui/Button";

const mentors = [
  {
    id: 1,
    name: "Jane Doe",
    avatar: "https://i.pravatar.cc/80?img=5",
    expertise: ["System Design", "Backend", "FAANG"],
    bio: "Ex-Google engineer. Passionate about helping you crack system design interviews.",
  },
  {
    id: 2,
    name: "John Smith",
    avatar: "https://i.pravatar.cc/80?img=8",
    expertise: ["DSA", "Algorithms", "Competitive Coding"],
    bio: "ICPC finalist. Loves teaching DSA and problem-solving strategies.",
  },
  {
    id: 3,
    name: "Priya Patel",
    avatar: "https://i.pravatar.cc/80?img=12",
    expertise: ["Behavioral", "HR", "Soft Skills"],
    bio: "HR at top startups. Can help you ace behavioral rounds and negotiations.",
  },
];

export default function MentorshipPage() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-50 to-indigo-100 py-12 px-4">
      <div className="max-w-4xl mx-auto">
        {/* Hero */}
        <div className="flex flex-col items-center mb-12">
          <h1 className="text-5xl font-extrabold text-indigo-700 mb-2 text-center drop-shadow">
            1:1 Mentorship
          </h1>
          <p className="text-lg text-gray-600 text-center mb-6 max-w-2xl">
            Connect with experienced mentors for personalized interview prep,
            career advice, and mock interviews. Level up with real guidance.
          </p>
          <Button
            size="lg"
            className="bg-gradient-to-r from-indigo-500 to-purple-400 text-white shadow-lg px-8 py-3 rounded-xl"
          >
            Become a Mentor
          </Button>
        </div>
        {/* Mentors List */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          {mentors.map((m) => (
            <div
              key={m.id}
              className="bg-white rounded-2xl shadow-md p-6 flex flex-col items-center justify-between hover:shadow-xl transition-shadow min-h-[400px]"
            >
              <img
                src={m.avatar}
                alt={m.name}
                className="w-20 h-20 rounded-full border-4 border-indigo-200 mb-4"
              />
              <h2 className="text-xl font-bold text-gray-800 mb-1">{m.name}</h2>
              <div className="flex flex-wrap gap-2 mb-2">
                {m.expertise.map((tag) => (
                  <span
                    key={tag}
                    className="bg-indigo-100 text-indigo-600 text-xs font-semibold px-2 py-1 rounded-lg"
                  >
                    {tag}
                  </span>
                ))}
              </div>
              <p className="text-gray-500 text-sm text-center mb-4 flex-1">
                {m.bio}
              </p>
              <Button
                size="sm"
                className="bg-indigo-500 text-white px-4 py-2 rounded-lg mt-4"
              >
                Book Session
              </Button>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
