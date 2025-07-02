import { Routes, Route } from "react-router-dom";
import { useState } from "react";

import Home from "./components/Home";
import Chatbot from "./components/Chatbot";

import { StarsCanvas } from "./components/canvas";

import AdminDashboard from "./components/admin_dashboard/AdminDashboard";
import Login from "./components/admin_dashboard/Login";
import Register from "./components/admin_dashboard/Register";
import AdminHome from "./components/admin_dashboard/AdminHome";

import ProtectedRoute from "./components/ProtectedRoutes";
import ForgetPassword from "./components/admin_dashboard/ForgetPassword";
import ForgetPasswordReset from "./components/admin_dashboard/ForgetPasswordReset";
import useInactivityLogout from "./components/UseInactivityLogout";
import useFavicon from "./useFavicon";

import InviteRegister from "./components/InviteRegister";

import InformBusiness from "./components/InformBusiness";

const LIGHT_ICON = '/logo.svg';
const DARK_ICON = '/logo_darker.svg';

function App() {
  useInactivityLogout();
  useFavicon(DARK_ICON, LIGHT_ICON);

  const [openSidebarToggle, setOpenSidebarToggle] = useState(false);

  const OpenSidebar = () => {
    setOpenSidebarToggle(!openSidebarToggle);
  };

  // return(
  //   <div className="p-4 text-black bg-red-500">
  //     Tailwind Test
  //   </div>
  // )

  return (
    <div className="w-full">
      <div className="absolute top-0 left-0 z-10 w-full h-full" >
        {/* <div className="z-1" > */}
        <Routes>
          <Route exact path="/" element={<InformBusiness />} />
          <Route exact path="/:businessSlug" element={<Home />} />
          <Route exact path="/register" element={<Register />} />
          <Route exact path="/login" element={<Login />} />
          <
          Route exact path="/request/password_reset" element={<ForgetPassword />} />
          <Route exact path="/password-reset/:token" element={<ForgetPasswordReset />} />
          <Route exact path="/adminHome" element={<AdminHome />} />
          <Route exact path="/chatbot/:businessSlug" element={<Chatbot />} />
          <Route path="/register/invite/:token" element={<InviteRegister />} />
          
          <Route element={<ProtectedRoute />}>
            <Route exact path="/admin" element={<AdminDashboard openSidebarToggle={openSidebarToggle} OpenSidebar={OpenSidebar} />} />
          </Route>
        </Routes>
      </div>
      <div className="relative z-0">
        <StarsCanvas />
      </div>
      {/* <div className="relative z-0">
          <RocketCanvas />
        </div> */}
    </div>
  )
}

export default App;
