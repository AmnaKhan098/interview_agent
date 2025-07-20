import { Button } from "../../../components/ui/Button";

const dummyStories = [
  {
    id: 1,
    name: "Jane Doe",
    avatar: "https://i.pravatar.cc/80?img=5",
    title: "From Rejection to FAANG Offer",
    story:
      "After months of failed interviews, I found the right resources and community support. Landed my dream job at Google!",
    role: "SWE @ Google",
  },
  {
    id: 2,
    name: "Rahul Singh",
    avatar: "https://i.pravatar.cc/80?img=8",
    title: "Cracked System Design Round",
    story:
      "System design always scared me. But with mock interviews and feedback, I aced it at Amazon.",
    role: "Backend Eng @ Amazon",
  },
  {
    id: 3,
    name: "Emily Chen",
    avatar: "https://i.pravatar.cc/80?img=12",
    title: "DSA Mastery in 3 Months",
    story:
      "Consistent practice and peer support helped me go from zero to hero in DSA. Got multiple offers!",
    role: "Intern @ Microsoft",
  },
];

export default function SuccessStoriesPage() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-pink-50 to-indigo-100 py-12 px-4">
      <div className="max-w-4xl mx-auto">
        {/* Hero Section */}
        <div className="flex flex-col items-center mb-12">
          <h1 className="text-5xl font-extrabold text-indigo-700 mb-3 text-center drop-shadow">
            Success Stories
          </h1>
          <p className="text-lg text-gray-600 text-center mb-6 max-w-2xl">
            Real journeys, real wins. Get inspired by community members who
            cracked their dream roles. Share your own story and motivate others!
          </p>
          <Button
            size="lg"
            className="bg-gradient-to-r from-pink-500 to-indigo-500 text-white shadow-lg px-8 py-3 rounded-xl hover:scale-105 transition-transform"
          >
            Share Your Story
          </Button>
        </div>
        {/* Stories List */}
        <div className="grid md:grid-cols-2 gap-8">
          {dummyStories.map((s) => (
            <div
              key={s.id}
              className="bg-white/80 rounded-3xl shadow-xl p-8 flex flex-col gap-4 border border-pink-100 hover:shadow-2xl transition-shadow relative overflow-hidden"
            >
              <div className="flex items-center gap-4 mb-2">
                <img
                  src={s.avatar}
                  alt={s.name}
                  className="w-16 h-16 rounded-full border-4 border-pink-200 shadow-md"
                />
                <div>
                  <div className="font-bold text-lg text-indigo-700">
                    {s.name}
                  </div>
                  <div className="text-sm text-pink-500 font-semibold">
                    {s.role}
                  </div>
                </div>
              </div>
              <div className="font-semibold text-xl text-gray-800 mb-1">
                {s.title}
              </div>
              <div className="text-gray-600 text-base mb-2">"{s.story}"</div>
              <div className="flex justify-end">
                <Button
                  variant="outline"
                  size="sm"
                  className="border-pink-300 text-pink-500 hover:bg-pink-50"
                >
                  Congratulate
                </Button>
              </div>
              <div className="absolute -top-8 -right-8 w-32 h-32 bg-pink-100 rounded-full opacity-30 blur-2xl z-0" />
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
