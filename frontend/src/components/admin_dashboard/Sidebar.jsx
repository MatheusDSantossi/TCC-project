import React from "react";
import {
  BsChatFill,
  BsGrid1X2Fill,
  BsFillArchiveFill,
  BsPeopleFill,
  BsListCheck,
  BsAlarm,
  BsJustify,
  BsXLg,
  BsAppIndicator,
  BsBookmark,
} from "react-icons/bs";

import logo from "../../assets/logo.svg";
import { MdLogout } from "react-icons/md";

import AxiosInstance from "../AxiosInstance";
import { useNavigate } from "react-router-dom";

import { AnimatePresence, motion } from "framer-motion";

import {
  Accordion,
  AccordionBody,
  AccordionHeader,
  Alert,
  Card,
  Chip,
  Drawer,
  IconButton,
  Input,
  List,
  ListItem,
  ListItemPrefix,
  ListItemSuffix,
  Typography,
} from "@material-tailwind/react";

function Sidebar({ openSidebarToggle, OpenSidebar, changeComponent }) {
  const navigate = useNavigate();

  const [open, setOpen] = React.useState(0);
  const [openAlert, setOpenAlert] = React.useState(true);
  const [isDrawerOpen, setIsDrawerOpen] = React.useState(false);

  const handleLogout = () => {
    AxiosInstance.post(`logoutall/`, {}).then(() => {
      // Clear user session data and navigate to login pagel
      localStorage.removeItem("Token");

      // Navigate to login page
      navigate("/login");
    });
  };

  const handleOpen = (value) => {
    setOpen(open === value ? 0 : value);
  };

  const openDrawer = () => setIsDrawerOpen(true);
  const closeDrawer = () => setIsDrawerOpen(false);

  return (
    <div className="absolute z-10 bg-primary">
      <IconButton variant="text" size="lg" onClick={openDrawer}>
        {isDrawerOpen ? (
          <BsXLg className="text-xl cursor-pointer hover:text-secondary" />
        ) : (
          <BsJustify className="text-xl cursor-pointer hover:text-secondary" />
        )}
      </IconButton>
      <Drawer open={isDrawerOpen} onClose={closeDrawer}>
        <Card
          color="transparent"
          shadow={false}
          className="h-[calc(100vh-2rem)] w-full p-4 bg-primary"
        >
          <div className="flex justify-center gap-4 p-4 mb-2 bg-primary">
            <img src={logo} alt="brand" className="size-28" />
          </div>
          <hr className="my-2 border-blue-gray-50" />
          <List>
            <ListItem>
              <ListItemPrefix
                className="flex p-2 cursor-pointer hover:text-primary hover:bg-secondary hover:rounded"
                onClick={(e) => {
                  changeComponent("dashboard");
                  closeDrawer();
                  e.preventDefault();
                }}
              >
                <BsGrid1X2Fill className="w-5 h-5 mr-3" />
                Dashaboard
              </ListItemPrefix>
              <ListItemSuffix>
                <Chip
                  value="14"
                  size="sm"
                  variant="ghost"
                  color="blue-gray"
                  className="rounded-full"
                />
              </ListItemSuffix>
            </ListItem>
            <ListItem>
              <ListItemPrefix
                onClick={(e) => {
                  changeComponent("productRegistration");
                  closeDrawer();
                  e.preventDefault();
                }}
                className="flex p-2 cursor-pointer hover:text-primary hover:bg-secondary hover:rounded"
              >
                <BsFillArchiveFill className="w-5 h-5 mr-3" />
                Products
              </ListItemPrefix>
            </ListItem>
            <ListItem>
              <ListItemPrefix
                onClick={(e) => {
                  changeComponent("chatbot");
                  closeDrawer();
                  e.preventDefault();
                }}
                className="flex p-2 cursor-pointer hover:text-primary hover:bg-secondary hover:rounded"
              >
                <BsChatFill className="w-5 h-5 mr-3" />
                Chatbot
              </ListItemPrefix>
            </ListItem>
            <ListItem>
              <ListItemPrefix onClick={handleLogout} className="flex p-2 cursor-pointer hover:text-primary hover:bg-red-500 hover:rounded">
                <MdLogout className="w-5 h-5 mr-3" />
                Log Out
              </ListItemPrefix>
            </ListItem>
          </List>
          <Alert
            open={openAlert}
            className="p-2 mt-auto bg-secondary text-primary"
            onClose={() => setOpenAlert(false)}
          >
            <BsBookmark className="w-12 h-12 mb-4" />
            <Typography variant="h6" className="mb-1">
              News
            </Typography>
            <Typography variant="small" className="font-normal opacity-80">
              Check the products to new edits.
            </Typography>
            <div className="flex gap-3 mt-4">
              <Typography
                as="a"
                href="#"
                variant="small"
                className="font-medium opacity-80"
                onClick={() => setOpenAlert(false)}
              >
                Dismiss
              </Typography>
              <Typography
                as="a"
                href="#"
                variant="small"
                className="font-medium"
              >
                Know more
              </Typography>
            </div>
          </Alert>
        </Card>
      </Drawer>
    </div>
  );
}

export default React.memo(Sidebar);
