import React from "react";
import { Trophy, Award, TrendingUp, CheckCircle, Clock } from "lucide-react";

const achievementsData = [
  {
    id: 1,
    title: "Login Streak King",
    description: "Maintained a login streak of 30 days!",
    icon: Clock,
    color: "text-blue-500",
    rank: "Diamond League",
    score: 3000,
  },
  {
    id: 2,
    title: "Problem Solver Pro",
    description: "Solved 100 coding challenges with 90%+ accuracy.",
    icon: CheckCircle,
    color: "text-green-500",
    rank: "Platinum Tier",
    score: 2500,
  },
  {
    id: 3,
    title: "Interview Ace",
    description: "Successfully completed 15 mock interviews.",
    icon: Award,
    color: "text-purple-500",
    rank: "Gold Standard",
    score: 2000,
  },
  {
    id: 4,
    title: "Rising Star",
    description: "Improved problem-solving speed by 20% in a month.",
    icon: TrendingUp,
    color: "text-yellow-500",
    rank: "Silver Performer",
    score: 1500,
  },
];

const AchievementCard = ({ achievement }: any) => {
  const Icon = achievement.icon;
  return (
    <div className="bg-white rounded-lg shadow-md p-6 flex flex-col items-center text-center">
      <Icon className={`w-12 h-12 mb-4 ${achievement.color}`} />
      <h3 className="text-xl font-semibold text-gray-800 mb-2">
        {achievement.title}
      </h3>
      <p className="text-gray-600 mb-4">{achievement.description}</p>
      <div className="mt-auto">
        <div className="text-sm font-medium text-gray-500">Your Rank:</div>
        <div className="text-2xl font-bold text-blue-600 mb-1">
          {achievement.rank}
        </div>
        <div className="text-md text-gray-700">Score: {achievement.score}</div>
      </div>
    </div>
  );
};

const AchievementsPage = () => {
  return (
    <div className="min-h-screen bg-gray-100 p-8">
      <div className="max-w-6xl mx-auto">
        <h1 className="text-4xl font-bold text-gray-900 mb-8 text-center">
          Your Achievements
        </h1>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {achievementsData.map((achievement) => (
            <AchievementCard key={achievement.id} achievement={achievement} />
          ))}
        </div>

        <section className="mt-12 p-6 bg-white rounded-lg shadow-md">
          <h2 className="text-3xl font-bold text-gray-800 mb-6 text-center">
            How Ranks Are Determined
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="flex flex-col items-center text-center">
              <Clock className="w-10 h-10 text-blue-500 mb-3" />
              <h3 className="text-xl font-semibold text-gray-700 mb-2">
                Login Streak
              </h3>
              <p className="text-gray-600">
                Ranks are based on consistent daily logins. The longer your
                streak, the higher your rank!
              </p>
            </div>
            <div className="flex flex-col items-center text-center">
              <CheckCircle className="w-10 h-10 text-green-500 mb-3" />
              <h3 className="text-xl font-semibold text-gray-700 mb-2">
                Problem Solving
              </h3>
              <p className="text-gray-600">
                Your rank reflects the number of problems solved, accuracy, and
                efficiency in coding challenges.
              </p>
            </div>
            <div className="flex flex-col items-center text-center">
              <Award className="w-10 h-10 text-purple-500 mb-3" />
              <h3 className="text-xl font-semibold text-gray-700 mb-2">
                Interview Performance
              </h3>
              <p className="text-gray-600">
                Achieve higher ranks by performing well in mock interviews and
                receiving positive feedback.
              </p>
            </div>
          </div>
        </section>
      </div>
    </div>
  );
};

export default AchievementsPage;
