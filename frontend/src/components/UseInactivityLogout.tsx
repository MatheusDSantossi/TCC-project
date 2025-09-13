import { useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { routes } from "../routes/routes";

const useInactivityLogout = (timeout = 7200000) => {
  // 7200000 = 2 hours
  // routes
  const adminHomeRoute = routes.admin.adminHome;

  // navigate
  const navigate = useNavigate();

  useEffect(() => {
    let timer;

    const resetTimer = () => {
      if (timer) clearTimeout(timer);
      timer = setTimeout(() => {
        localStorage.removeItem("Token");
        navigate(adminHomeRoute);
      }, timeout);
    };

    // List of events that indicate activity
    const events = [
      "mousemove",
      "keypress",
      "mousedown",
      "touchstart",
      "scroll",
      "keydown",
    ];

    events.forEach((event) => window.addEventListener(event, resetTimer));

    // Initialize the timer
    resetTimer();

    return () => {
      clearTimeout(timer);
      events.forEach((event) => window.removeEventListener(event, resetTimer));
    };
  }, [navigate, timeout]);
};

export default useInactivityLogout;
