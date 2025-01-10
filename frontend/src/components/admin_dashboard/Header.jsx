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

  // useEffect(() => {

  //   getUserName();

  //   return () => {

  //   };
  // }, []);

  const getUserName = () => {
    AxiosInstance.get("http://127.0.0.1:8000/employee_details/")

      .then((response) => {
        if (!response.ok) {
          throw new Error("Failed to fetch user details");
        }
        return response.json();
      })
      .then((data) => {
        if (data.employee_name) {
          const userName = data.employee_name;
          setUsername(userName);
          console.log(userName);
        } else {
          // window.location.href = "/login";
          console.error("No user logged in");
        }
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
      <div className="flex items-center space-x-4">
        <span className="text-xl cursor-pointer hover:text-secondary">
          <BsFillBellFill />
        </span>
        <span className="flex text-xl cursor-pointer hover:text-secondary">
          <BsPersonCircle /> {userName}
        </span>
      </div>
    </header>
  );
}

export default Header;
