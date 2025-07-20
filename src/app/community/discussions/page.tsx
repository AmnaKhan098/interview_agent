import { Button } from "../../../components/ui/Button";

const dummyDiscussions = [
  {
    id: 1,
    title: "How to prepare for system design interviews?",
    author: "Alice",
    avatar: "https://i.pravatar.cc/40?img=1",
    tags: ["System Design", "Interview"],
    replies: 12,
    lastReply: "2h ago",
  },
  {
    id: 2,
    title: "Best resources for DSA practice?",
    author: "Bob",
    avatar: "https://i.pravatar.cc/40?img=2",
    tags: ["DSA", "Resources"],
    replies: 8,
    lastReply: "30m ago",
  },
  {
    id: 3,
    title: "Behavioral round: what to expect?",
    author: "Carol",
    avatar: "https://i.pravatar.cc/40?img=3",
    tags: ["Behavioral", "HR"],
    replies: 5,
    lastReply: "10m ago",
  },
];

export default function DiscussionsPage() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 py-12 px-4">
      <div className="max-w-3xl mx-auto">
        {/* Hero */}
        <div className="flex flex-col items-center mb-10">
          <h1 className="text-5xl font-extrabold text-indigo-700 mb-2 text-center drop-shadow">
            Community Discussions
          </h1>
          <p className="text-lg text-gray-600 text-center mb-6">
            Ask questions, share tips, and help each other ace interviews.
          </p>
          <Button
            size="lg"
            className="bg-gradient-to-r from-indigo-500 to-blue-400 text-white shadow-lg px-8 py-3 rounded-xl"
          >
            Start Discussion
          </Button>
        </div>
        {/* Discussions List */}
        <div className="space-y-6">
          {dummyDiscussions.map((d) => (
            <div
              key={d.id}
              className="bg-white rounded-2xl shadow-md p-6 flex items-center gap-6 hover:shadow-xl transition-shadow"
            >
              <img
                src={d.avatar}
                alt={d.author}
                className="w-14 h-14 rounded-full border-2 border-indigo-200"
              />
              <div className="flex-1">
                <div className="flex items-center gap-2 mb-1">
                  {d.tags.map((tag) => (
                    <span
                      key={tag}
                      className="bg-indigo-100 text-indigo-600 text-xs font-semibold px-2 py-1 rounded-lg"
                    >
                      {tag}
                    </span>
                  ))}
                </div>
                <h2 className="text-xl font-bold text-gray-800 mb-1">
                  {d.title}
                </h2>
                <div className="text-sm text-gray-500">
                  By <span className="font-medium">{d.author}</span> · Last
                  reply {d.lastReply}
                </div>
              </div>
              <div className="flex flex-col items-center">
                <span className="text-indigo-600 font-bold text-lg">
                  {d.replies}
                </span>
                <span className="text-xs text-gray-400">Replies</span>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
