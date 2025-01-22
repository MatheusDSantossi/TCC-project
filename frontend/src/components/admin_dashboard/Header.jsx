import { useEffect, useState } from "react";
import {
  BsFillBellFill,
  BsPersonCircle,
  BsArrowDownRightSquare,
  BsArrowDownSquare,
} from "react-icons/bs";

import { businessName } from "../../utils/BusinessInfo";

function Header() {
  const [userName, setUsername] = useState("");
  const [isClicked, setIsClicked] = useState(false);

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
    <header className="flex items-center justify-end p-5 shadow-md bg-primary">
      <div className="flex items-center justify-around space-x-4">
        <span className="mr-20 text-lg font-bold">{businessName}</span>
        <span className="flex text-xl cursor-pointer hover:text-secondary">
          <BsFillBellFill />
          <span className="text-xs">3</span>
        </span>

        <div className="relative">
          <span
            className="flex items-center text-xl cursor-pointer hover:text-secondary"
            onClick={toggleDropdown}
          >
            <BsPersonCircle className="mr-2" />
            {isClicked ? <BsArrowDownSquare /> : <BsArrowDownRightSquare />}
          </span>

          {/* Dropdown Menu */}
          {isClicked && (
            <div className="absolute right-0 p-4 mt-2 transition-all duration-300 ease-in-out transform rounded-lg shadow-md bg-tertiary">
              <p
                className="text-sm font-medium"
              >
                <span className="font-bold">Logged in as:</span> {userName}
              </p>
            </div>
          )}
        </div>
      </div>
    </header>
  );
}

export default Header;
