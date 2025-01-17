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
      <input
        type="text"
        name="search"
        placeholder="Type to search..."
        className="w-1/3 p-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-secondary"
      />
      <div className="flex items-center space-x-4 justify-center">
        <span className="text-xl cursor-pointer hover:text-secondary">
          <BsFillBellFill />
        </span>
        <span className="flex text-xl cursor-pointer hover:text-secondary items-center">
          <BsPersonCircle className="mr-2" /> {userName}
        </span>
      </div>
    </header>
  );
}

export default Header;
