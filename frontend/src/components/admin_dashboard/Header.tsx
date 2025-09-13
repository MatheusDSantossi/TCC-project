import { useEffect, useState } from "react";

import AxiosInstance from "../AxiosInstance";
import { MdLogout } from "react-icons/md";
import { useNavigate } from "react-router-dom";

import { businessName } from "../../utils/BusinessInfo";
import { ListItemPrefix } from "@material-tailwind/react";

import * as motion from "motion/react-client";
import { AnimatePresence } from "motion/react";

import "../../App.css";
import { routes } from "../../routes/routes";

function Header() {
  const [userName, setUsername] = useState("");
  const [isClicked, setIsClicked] = useState(false);
  const navigate = useNavigate();

  const businessSlug = localStorage.getItem("BusinessSlug");

  const handleLogout = () => {
    // routes
    const adminHomeRoute = routes.admin.adminHome;

    AxiosInstance.post(`logoutall/`, {}).then(() => {
      // Clear user session data and navigate to login pagel
      localStorage.removeItem("Token");

      // Navigate to home page
      navigate(adminHomeRoute);
    });
  };

  useEffect(() => {
    getUserName();
  }, []);

  const getUserName = () => {
    const userName = localStorage.getItem("Email") || "Guest";
    setUsername(userName);
    console.log("Username: ", userName);
  };

  const toggleDropdown = () => {
    setIsClicked((prev) => !prev);
  };

  return (
    <header className="flex items-center justify-between p-5 drop-shadow-3xl bg-primary">
      {/* <div className="flex items-center justify-around space-x-4"> */}
      <span className="absolute mr-20 text-lg font-semibold transform -translate-x-1/2 left-1/2">
        Bem-vindo ao {businessName} ({businessSlug})
      </span>
      {/* Icons container */}
      <span className="flex items-center ml-auto space-x-4 text-xl cursor-pointer hover:text-secondary">
        <div className="flex">
          {/* <BsFillBellFill />
        <span className="text-xs">3</span> */}
        </div>
      </span>

      <div className="relative ml-4">
        <motion.span
          className="flex items-center text-xl rounded-full cursor-pointer"
          onClick={toggleDropdown}
          whileTap={{ y: 1 }}
        >
          <p className="p-2 rounded-full text-primary bg-secondary hover:bg-secondary-dark">
            {userName.slice(0, 2).toUpperCase()}
          </p>
          {/* {isClicked ? <BsArrowDownSquare /> : <BsArrowDownRightSquare />} */}
        </motion.span>

        <AnimatePresence initial={false}>
          {isClicked ? (
            <motion.div
              className="absolute right-0 z-10 justify-end p-4 mt-2 text-black transition-all duration-300 ease-in-out transform rounded-lg shadow-md bg-secondary"
              initial={{ opacity: 0, scale: 0.5 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0 }}
              transition={{
                duration: 0.1,
                delay: 0.1,
                ease: "easeInOut",
                type: "tween",
              }}
            >
              <p className="text-sm font-medium">
                <span className="font-semibold">Logado como:</span> {userName}
                <ListItemPrefix
                  className="flex justify-center p-2 mt-4 cursor-pointer hover:bg-red-500 hover:rounded"
                  onClick={handleLogout}
                >
                  <MdLogout className="w-5 h-5 mr-3" />
                  Sair
                </ListItemPrefix>
              </p>
            </motion.div>
          ) : null}
        </AnimatePresence>
      </div>
      {/* </div> */}
    </header>
  );
}

export default Header;
