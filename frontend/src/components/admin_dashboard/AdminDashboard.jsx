import { useState } from "react";
import Header from "./Header";
import Sidebar from "./Sidebar";
import HomeDashboard from "./HomeDashboard";

function AdminDashboard({ openSidebarToggle, OpenSidebar }) {

    return (
        <div className="flex h-screen bg-white-100">
            <Sidebar openSidebarToggle={openSidebarToggle} OpenSidebar={OpenSidebar} />
            <div className="flex flex-col flex-1">
                <Header openSidebarToggle={openSidebarToggle} OpenSidebar={OpenSidebar} />
                <HomeDashboard />
            </div>
        </div>
    );
}

export default AdminDashboard;
