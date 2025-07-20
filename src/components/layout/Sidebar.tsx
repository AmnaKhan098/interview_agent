"use client";
// @ts-nocheck
import React from "react";
import { motion, AnimatePresence } from "framer-motion";
import { useUser } from "@clerk/nextjs";
import {
  Home,
  BookOpen,
  BarChart3,
  ChevronDown,
  Trophy,
  Target,
  Calendar,
  MessageSquare,
  X,
  LogOut,
} from "lucide-react";
import type { Variants } from "framer-motion";
import { useRouter } from "next/navigation";

interface SidebarProps {
  isOpen: boolean;
  onClose: () => void;
}

const Sidebar = ({ isOpen, onClose }: SidebarProps) => {
  const router = useRouter();
  const [activeItem, setActiveItem] = React.useState("dashboard");
  const [expandedMenus, setExpandedMenus] = React.useState<Record<string, boolean>>({});

  const { isLoaded, isSignedIn, user } = useUser();
  const userName = user?.fullName || "Guest";
  const userEmail = user?.primaryEmailAddress?.emailAddress || "guest@example.com";

  const toggleMenu = (menuId: string) => {
    setExpandedMenus((prev) => ({
      ...prev,
      [menuId]: !prev[menuId],
    }));
  };

  const navigationItems = [
    {
      id: "dashboard",
      label: "Dashboard",
      icon: Home,
      href: "/dashboard",
      badge: null,
    },
    {
      id: "practice",
      label: "Practice",
      icon: BookOpen,
      href: "/practice",
      badge: "New",
      submenu: [
        { id: "mock-interview", label: "Mock Interview", href: "/practice/mock" },
        { id: "coding-challenges", label: "Coding Challenges", href: "/practice/coding" },
        { id: "behavioral", label: "Behavioral Questions", href: "/practice/behavioral" },
      ],
    },
    {
      id: "progress",
      label: "Progress",
      icon: BarChart3,
      href: "/progress",
      badge: null,
    },
  ];

  const quickActions = [
    { id: "achievements", label: "Achievements", icon: Trophy, color: "text-yellow-500" },
    { id: "goals", label: "Goals", icon: Target, href: "/goals", color: "text-blue-500" },
    { id: "schedule", label: "Schedule", icon: Calendar, color: "text-green-500" },
    { id: "feedback", label: "Feedback", icon: MessageSquare, color: "text-purple-500" },
  ];

  const sidebarVariants: Variants = {
    open: { x: 0, transition: { type: "spring", stiffness: 300, damping: 30 } },
    closed: { x: "-100%", transition: { type: "spring", stiffness: 300, damping: 30 } },
  };

  const overlayVariants = {
    open: { opacity: 0.5, display: "block" },
    closed: { opacity: 0, transitionEnd: { display: "none" } },
  };

  const NavigationItem = ({ item, isActive, onClick }: any) => {
    const Icon = item.icon;
    const hasSubmenu = item.submenu?.length > 0;
    const isExpanded = expandedMenus[item.id];

    return (
      <div className="mb-1">
        <motion.div
          className={`flex items-center justify-between p-3 rounded-xl cursor-pointer transition-all duration-200 ${
            isActive
              ? "bg-gradient-to-r from-blue-500 to-purple-600 text-white shadow-lg"
              : "text-gray-700 hover:bg-gray-100 hover:text-blue-600"
          }`}
          onClick={() => {
            onClick(item.id);
            if (item.href && !hasSubmenu) {
              router.push(item.href);
              return;
            }
            if (hasSubmenu) toggleMenu(item.id);
          }}
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
        >
          <div className="flex items-center space-x-3">
            <Icon className="w-5 h-5" />
            <span className="font-medium">{item.label}</span>
          </div>

          <div className="flex items-center space-x-2">
            {item.badge && (
              <span className="px-2 py-1 text-xs rounded-full bg-green-100 text-green-700">
                {item.badge}
              </span>
            )}
            {hasSubmenu && (
              <ChevronDown className={`w-4 h-4 transition-transform ${isExpanded ? "rotate-180" : ""}`} />
            )}
          </div>
        </motion.div>

        <AnimatePresence>
          {hasSubmenu && isExpanded && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: "auto" }}
              exit={{ opacity: 0, height: 0 }}
              transition={{ duration: 0.2 }}
              className="ml-8 mt-2 space-y-1"
            >
              {item.submenu.map((subItem: any) => (
                <motion.div
                  key={subItem.id}
                  className="p-2 text-sm text-gray-600 hover:text-blue-600 hover:bg-blue-50 rounded-lg cursor-pointer transition-colors"
                  onClick={() => {
                    setActiveItem(subItem.id);
                    if (subItem.href) router.push(subItem.href);
                  }}
                  whileHover={{ x: 4 }}
                >
                  {subItem.label}
                </motion.div>
              ))}
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    );
  };

  return (
    <>
      {/* Backdrop */}
      <motion.div
        className="fixed inset-0 bg-black z-40"
        initial="closed"
        animate={isOpen ? "open" : "closed"}
        variants={overlayVariants}
        onClick={onClose}
      />

      {/* Sidebar */}
      <motion.aside
        className="fixed top-0 left-0 h-screen bg-white z-50 shadow-2xl"
        initial="closed"
        animate={isOpen ? "open" : "closed"}
        variants={sidebarVariants}
        style={{ width: "280px" }}
      >
        <div className="flex flex-col h-full">
          {/* Header */}
          <div className="flex items-center justify-between p-4 border-b">
            <div className="flex items-center space-x-3">
              <div className="w-8 h-8 bg-gradient-to-r from-blue-500 to-purple-600 rounded-lg flex items-center justify-center">
                <span className="text-white font-bold">IP</span>
              </div>
              <span className="font-semibold text-gray-800 cursor-pointer" onClick={() => router.push("/")}>
                Interview Prep
              </span>
            </div>
            <button onClick={onClose} className="p-2 hover:bg-gray-100 rounded-lg transition-colors">
              <X className="w-5 h-5 text-gray-500" />
            </button>
          </div>

          {/* Navigation */}
          <nav className="flex-1 overflow-y-auto p-4">
            {navigationItems.map((item) => (
              <NavigationItem key={item.id} item={item} isActive={activeItem === item.id} onClick={setActiveItem} />
            ))}
          </nav>

          {/* Quick Actions */}
          <div className="p-4 border-t">
            <div className="grid grid-cols-2 gap-2">
              {quickActions.map((action) => (
                <motion.button
                  key={action.id}
                  className="flex flex-col items-center p-3 rounded-xl bg-gray-50 hover:bg-gray-100 transition-colors"
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  onClick={() => {
                    if (action.id === "feedback") router.push("/feedback/all-feedbacks");
                    if (action.id === "goals") router.push("/goals");
                    if (action.id === "schedule") router.push("/schedule");
                    if (action.id === "achievements") router.push("/achievements");
                  }}
                >
                  <action.icon className={`w-5 h-5 ${action.color}`} />
                  <span className="text-xs mt-1 text-gray-600">{action.label}</span>
                </motion.button>
              ))}
            </div>
          </div>

          {/* User Info and Logout */}
          <div className="mt-auto p-4 border-t border-gray-200">
            <div className="flex items-center space-x-3 mb-4">
              <div className="w-10 h-10 bg-blue-200 rounded-full flex items-center justify-center text-blue-700 font-bold">
                {userName.charAt(0).toUpperCase()}
              </div>
              <div>
                <p className="font-semibold text-gray-800">{userName}</p>
                <p className="text-sm text-gray-500">{userEmail}</p>
              </div>
            </div>
            <button className="w-full flex items-center justify-center space-x-2 p-3 rounded-xl bg-red-500 text-white hover:bg-red-600 transition-colors duration-200">
              <LogOut className="w-5 h-5" />
              <span>Logout</span>
            </button>
          </div>
        </div>
      </motion.aside>
    </>
  );
};

export default Sidebar;
