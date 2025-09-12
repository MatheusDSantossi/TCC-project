import { Routes, Route } from "react-router-dom";
import { useState } from "react";

import Home from "./components/Home";
import Chatbot from "./components/Chatbot";

import { StarsCanvas } from "./components/canvas/index";

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
import { routes } from "./routes/routes";

const LIGHT_ICON = '/logo.svg';
const DARK_ICON = '/logo_darker.svg';

function App() {
  // routes
  const adminHomeRoute = routes.admin.adminHome;
  const adminMainRoute = routes.admin.adminMain;
  const registerRoute = routes.form.register;
  const loginRoute = routes.form.login;
  const requestPasswordRoute = routes.form.requestPasswordReset;
  const requestPasswordTokenRoute = routes.form.passwordResetToken;
  const chatbotMainRoute = routes.chatbot.chatbotMain;
  const invitationRegisterRoute = routes.form.invitationRegister;
  const businessSlugRoute = routes.business.businessSlug;

  useInactivityLogout();
  useFavicon(DARK_ICON, LIGHT_ICON);

  const [openSidebarToggle, setOpenSidebarToggle] = useState(false);

  const OpenSidebar = () => {
    setOpenSidebarToggle(!openSidebarToggle);
  };
  
  return (
    <div className="w-full">
      <div className="absolute top-0 left-0 z-10 w-full" >
        {/* <div className="z-1" > */}
        <Routes>
          <Route path="/" element={<InformBusiness />} />
          <Route path={businessSlugRoute} element={<Home />} />
          <Route path={registerRoute} element={<Register />} />
          <Route path={loginRoute} element={<Login />} />
          <Route path={requestPasswordRoute} element={<ForgetPassword />} />
          <Route path={requestPasswordTokenRoute} element={<ForgetPasswordReset />} />
          <Route path={adminHomeRoute} element={<AdminHome />} />
          <Route path={chatbotMainRoute} element={<Chatbot />} />
          <Route path={invitationRegisterRoute} element={<InviteRegister />} />
          
          <Route element={<ProtectedRoute />}>
            <Route path={adminMainRoute} element={<AdminDashboard openSidebarToggle={openSidebarToggle} OpenSidebar={OpenSidebar} />} />
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
