import { useEffect, useState } from "react";
import Header from "./Header";
import Sidebar from "./Sidebar";
import HomeDashboard from "./HomeDashboard";
import ProductRegistration from "../ProductRegistration";

function AdminDashboard({ openSidebarToggle, OpenSidebar }) {

    const [currentComponent, setCurrentComponent] = useState("dashboard");

    // Sync state with the current URL path
    useEffect(() => {
        const handleLocationChange = () => {
            const path = window.location.pathname.replace("/", "") || window.location.hash.replace("#", "") || "dashboard";
            setCurrentComponent(path);
            console.log(currentComponent)
        }

        handleLocationChange();

        window.addEventListener("popstate", handleLocationChange);

        return () => window.removeEventListener("popstate", handleLocationChange);
    }, []);

    const renderComponent = () => {
        switch (currentComponent) {
            case "dashboard":
                return <HomeDashboard />;
            case "productRegistration":
                return <ProductRegistration />;
            // case "another":
            //     return <AnotherComponent />;
            //...
            default:
                // return <HomeDashboard />;
                return <div>Not found!!!!</div>;
        }
    };

    const changeComponent = (component) => {
        window.history.pushState({}, "", `/${component}`);
        setCurrentComponent(component);
    };

    return (
            <div className="flex h-screen bg-white-100">
                <Sidebar openSidebarToggle={openSidebarToggle} OpenSidebar={OpenSidebar} changeComponent={changeComponent} />
                <div className="flex flex-col flex-1">
                    <Header openSidebarToggle={openSidebarToggle} OpenSidebar={OpenSidebar} />
                    {renderComponent()}                    
                </div>
            </div>
    );

}

export default AdminDashboard;
