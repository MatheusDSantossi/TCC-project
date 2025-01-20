import React, { useEffect, useState } from "react";
import {
  BsFillBellFill,
  BsFillEnvelopeFill,
  BsPersonCircle,
  BsSearch,
  BsJustify,
  BsX,
  BsXLg,
} from "react-icons/bs";

import AxiosInstance from "../AxiosInstance";

function Header({ OpenSidebar, openSidebarToggle }) {
  const [userName, setUsername] = useState("");

  useEffect(() => {
    getUserName();
  }, []);

  const getUserName = () => {
    AxiosInstance.get(`users/`)

      .then((response) => {

        // for (const user of response.data) {
        //   console.log("User: ", user['last_login']);
        //   if (user['last_login']) {
        //     const userName = user['email'];
        //     setUsername(userName);
        //     console.log("Username: ", userName);
        //   } else {
        //     // window.location.href = "/login";
        //     console.error("No user logged in");
        //   }
        // }

        const userName = localStorage.getItem("Email");
        setUsername(userName);
        console.log("Username: ", userName);

      })
      .catch((error) => console.error("Error getting the data: ", error));
    // window.location.href = "/login";
  };

  return (
    <header className="flex items-center justify-between p-4 shadow-md bg-primary">
      <div className="menu-icon">
        {openSidebarToggle ? (
          <BsXLg
            className="text-xl cursor-pointer hover:text-secondary"
            onClick={OpenSidebar}
          />
        ) : (
          <BsJustify
            className="text-xl cursor-pointer hover:text-secondary"
            onClick={OpenSidebar}
          />
        )}
      </div>

      <div className="flex items-center justify-center space-x-4">
        <span className="flex text-xl cursor-pointer hover:text-secondary">
          <BsFillBellFill />
          <span className="text-xs">3</span>
        </span>
        <span className="flex items-center text-xl cursor-pointer hover:text-secondary">
          <BsPersonCircle className="mr-2" /> {userName}
        </span>
      </div>
    </header>
  );
}

export default Header;
