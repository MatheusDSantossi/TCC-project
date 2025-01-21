import { useEffect, useState } from "react";
import Header from "./Header";
import Sidebar from "./Sidebar";
import HomeDashboard from "./HomeDashboard";
import ProductRegistration from "../ProductRegistration";
import Chatbot from "../Chatbot";

import UserList from "./UsersList";

import "../../App.css";

function AdminDashboard() {
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);

  const [currentComponent, setCurrentComponent] = useState("dashboard");

  const handleSibedarToggle = (isOpen) => {
    setIsSidebarOpen(isOpen);
  };

  // Sync state with the current URL path
  useEffect(() => {
    window.addEventListener("beforeunload", alertUser);

    const handleLocationChange = () => {
      const path =
        window.location.pathname.replace("/", "") ||
        window.location.hash.replace("#", "") ||
        "dashboard";
      setCurrentComponent(path);
      console.log(currentComponent);
    };

    handleLocationChange();

    window.addEventListener("popstate", handleLocationChange);

    return () => {
      window.removeEventListener("popstate", handleLocationChange);
      window.removeEventListener("beforeunload", alertUser);
    };
  }, []);

  const alertUser = (e) => {
    e.preventDefault();
    return "Are you sure you want to leave this page?";
  };

  // Render a login screen before the admin screen

  const renderComponent = () => {
    switch (currentComponent) {
      case "dashboard":
        return <HomeDashboard />;
      case "productRegistration":
        return <ProductRegistration />;
      case "chatbot":
        return <Chatbot />;
      case "userList":
        return <UserList />;

      default:
        return <HomeDashboard />;
      // return <div>Not found!!!!</div>;
    }
  };

  const changeComponent = (component) => {
    // this method
    // window.history.pushState({}, "", `/${component}`);
    setCurrentComponent(component);
  };

  return (
    <div className="flex w-full h-screen bg-white-100">
      {isSidebarOpen && <div className="bg-primary"></div>}
      <Sidebar
        onSidebarToggle={handleSibedarToggle}
        changeComponent={changeComponent}
      />

      <div
        className={`flex flex-col flex-1 transition-all ${
          isSidebarOpen ? "bg-overlay" : ""
        }`}
      >
        {/* Overlay Background */}
        {isSidebarOpen && (
          <div
            className="fixed top-0 left-0 z-50 w-full h-screen bg-black opacity-50"
            onClick={() => setIsSidebarOpen(false)}
          />
        )}
        <Header />
        {renderComponent()}
      </div>
    </div>
  );
}

export default AdminDashboard;
