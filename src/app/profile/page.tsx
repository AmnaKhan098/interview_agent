"use client";
import { useAuth } from "@/src/lib/contexts/AuthContext";
import { useState, useEffect } from "react";

export default function ProfilePage() {
  const { user, updateUserProfile } = useAuth();
  const [editing, setEditing] = useState(false);
  const [form, setForm] = useState({
    name: "",
    email: "",
    title: "",
    location: "",
    bio: "",
    avatar: "",
  });
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");

  // Initialize form with user data when it's available
  useEffect(() => {
    if (user) {
      setForm({
        name: user.name || "",
        email: user.email || "",
        title: user.title || "",
        location: user.location || "",
        bio: user.bio || "",
        avatar: user.avatar || "",
      });
    }
  }, [user]);

  const handleChange = (
    e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>
  ) => {
    setForm({ ...form, [e.target.name]: e.target.value });
  };

  const handleSave = async () => {
    setIsLoading(true);
    setError("");
    setSuccess("");

    try {
      await updateUserProfile({
        name: form.name,
        title: form.title,
        location: form.location,
        bio: form.bio,
        avatar: form.avatar,
      });
      setEditing(false);
      setSuccess("Profile updated successfully");
    } catch (err) {
      setError((err as any).message || "Failed to update profile");
    } finally {
      setIsLoading(false);
    }
  };

  if (!user) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 to-purple-100 flex items-center justify-center py-16 px-4">
        <div className="bg-white shadow-xl rounded-3xl p-10 max-w-xl w-full animate-pulse">
          <p className="text-center text-gray-500">Loading profile...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-purple-100 flex items-center justify-center py-16 px-4">
      <div className="bg-white shadow-2xl rounded-3xl p-10 max-w-xl w-full animate-fade-in border border-gray-100">
        {success && (
          <div className="mb-6 bg-green-50 border-l-4 border-green-500 p-4 rounded-md">
            <p className="text-green-700">{success}</p>
          </div>
        )}

        {error && (
          <div className="mb-6 bg-red-50 border-l-4 border-red-500 p-4 rounded-md">
            <p className="text-red-700">{error}</p>
          </div>
        )}

        <div className="flex flex-col items-center mb-8">
          <div className="relative">
            <img
              src={
                user.avatar ||
                `https://ui-avatars.com/api/?name=${encodeURIComponent(
                  user.name
                )}&background=random`
              }
              alt="avatar"
              className="w-36 h-36 rounded-full border-4 border-purple-400 shadow-lg object-cover"
            />
            <span className="absolute bottom-2 right-2 bg-green-400 border-2 border-white w-6 h-6 rounded-full block"></span>
          </div>
          <h2 className="mt-4 text-4xl font-bold text-gray-800">{user.name}</h2>
          <p className="text-purple-600 font-medium text-lg">
            {user.title || "No title set"}
          </p>
          <p className="text-gray-500 text-base">
            {user.location || "No location set"}
          </p>
        </div>
        <div className="mb-8 text-center">
          <p className="text-gray-700 italic text-lg">
            {user.bio || "No bio set"}
          </p>
        </div>
        {editing ? (
          <div className="space-y-4">
            <input
              className="w-full px-4 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-400"
              name="name"
              value={form.name}
              onChange={handleChange}
              placeholder="Name"
            />
            <input
              className="w-full px-4 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-400"
              name="title"
              value={form.title}
              onChange={handleChange}
              placeholder="Title"
            />
            <input
              className="w-full px-4 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-400"
              name="location"
              value={form.location}
              onChange={handleChange}
              placeholder="Location"
            />
            <textarea
              className="w-full px-4 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-400"
              name="bio"
              value={form.bio}
              onChange={handleChange}
              placeholder="Bio"
              rows={3}
            />
            <input
              className="w-full px-4 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-400"
              name="avatar"
              value={form.avatar}
              onChange={handleChange}
              placeholder="Avatar URL"
            />
            <div className="flex justify-end space-x-2">
              <button
                className="px-4 py-2 rounded-lg bg-gray-200 text-gray-700 hover:bg-gray-300"
                onClick={() => setEditing(false)}
                disabled={isLoading}
              >
                Cancel
              </button>
              <button
                className="px-4 py-2 rounded-lg bg-gradient-to-r from-blue-500 to-purple-600 text-white font-semibold shadow-md hover:from-blue-600 hover:to-purple-700 disabled:opacity-50"
                onClick={handleSave}
                disabled={isLoading}
              >
                {isLoading ? "Saving..." : "Save"}
              </button>
            </div>
          </div>
        ) : (
          <div className="flex flex-col items-center">
            <div className="w-full flex flex-col items-center space-y-2 mb-4">
              <span className="text-gray-600 text-base">
                <b>Email:</b> {user.email}
              </span>
              {user.title && (
                <span className="text-gray-600 text-base">
                  <b>Title:</b> {user.title}
                </span>
              )}
              {user.location && (
                <span className="text-gray-600 text-base">
                  <b>Location:</b> {user.location}
                </span>
              )}
            </div>
            <button
              className="px-6 py-2 rounded-lg bg-gradient-to-r from-blue-500 to-purple-600 text-white font-semibold shadow-md hover:from-blue-600 hover:to-purple-700 transition-all"
              onClick={() => setEditing(true)}
            >
              Edit Profile
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
