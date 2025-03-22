import { useEffect, useState } from "react";
import { Outlet, Navigate } from "react-router-dom";

const ProtectedRoute = () => {
    // const [isAuthenticated, setIsAuthenticated] = useState(false);

    // useEffect(() => {
    //     const token = localStorage.getItem('token');
    //     const checkAuth = () => {
    //         setIsAuthenticated(!!token);
    //     }

    //     // Listen for storage changes
    //     window.addEventListener("storage", checkAuth)

    //     return () => {
    //         window.removeEventListener("storage", checkAuth)
    //     }

    // }, []);

    // return isAuthenticated? (
    //     <Outlet />
    // ) : (
    //     <Navigate to="/login" />
    // );

    // Check if user is authenticated
    // If not, redirect to login page
    const token = localStorage.getItem('Token');
    // Check the expeiry of the token to make it more securitly
    return (
        token ? (
            <Outlet />
        ) : (
            <Navigate to="/adminHome" />
        )
    )
};

export default ProtectedRoute;
