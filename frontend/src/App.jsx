import { BrowserRouter, Routes, Route } from "react-router-dom";
import { useEffect, useState } from "react";

import Home from "./components/Home";
import Chatbot from "./components/Chatbot";
import ProductRegistration from "./components/ProductRegistration";

import { RocketCanvas, StarsCanvas } from "./components/canvas";

import AdminDashboard from "./components/admin_dashboard/AdminDashboard";

function App() {
  const [openSidebarToggle, setOpenSidebarToggle] = useState(false);

  const OpenSidebar = () => {
    setOpenSidebarToggle(!openSidebarToggle);
  };

  return (
    <div>
      <BrowserRouter>
        {/* <div className="relative z-10" > */}
        <div className="" >
          <Routes>
            <Route exact path="/" element={<Home />} />
            <Route exact path="/chatbot" element={<Chatbot />} />
            <Route exact path="/productregistration" element={<ProductRegistration />} />
            <Route exact path="/admin" element={<AdminDashboard openSidebarToggle={openSidebarToggle} OpenSidebar={OpenSidebar} />} />
          </Routes>
        </div>
        {/* <div className="relative z-0">
          <StarsCanvas />
        </div>
        <div className="relative z-0">
          <RocketCanvas />
        </div> */}
      </BrowserRouter>

    </div>
  )
}

export default App;
