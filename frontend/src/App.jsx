import { BrowserRouter, Routes, Route } from "react-router-dom";
import { useEffect, useState } from "react";

import Home from "./components/Home";
import Chatbot from "./components/Chatbot";
import ProductRegistration from "./components/ProductRegistration";

import { RocketCanvas, StarsCanvas } from "./components/canvas";

import AdminDashboard from "./components/admin_dashboard/AdminDashboard";
import Login from "./components/admin_dashboard/Login";
import Register from "./components/admin_dashboard/Register";
import AdminHome from "./components/admin_dashboard/AdminHome";

function App() {
  const [openSidebarToggle, setOpenSidebarToggle] = useState(false);

  const OpenSidebar = () => {
    setOpenSidebarToggle(!openSidebarToggle);
  };

  return (
    <div>
      <BrowserRouter>
        <div className="relative z-10" >
        {/* <div className="z-1" > */}
          <Routes>
            <Route exact path="/" element={<Home />} />
            <Route exact path="/chatbot" element={<Chatbot />} />
            <Route exact path="/login" element={<Login />} />
            <Route exact path="/register" element={<Register />} />
            <Route exact path="/adminHome" element={<AdminHome />} />
            <Route exact path="/admin" element={<AdminDashboard openSidebarToggle={openSidebarToggle} OpenSidebar={OpenSidebar} />} />
          </Routes>
        </div>
        <div className="relative z-0">
          <StarsCanvas />
        </div>
        {/* <div className="relative z-0">
          <RocketCanvas />
        </div> */}
      </BrowserRouter>

    </div>
  )
}

export default App;
