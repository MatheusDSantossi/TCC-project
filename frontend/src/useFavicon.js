import { useEffect } from 'react';

//**
// @param {string} lightHref absolute path to the light-mode icon
// @param {string} darkHref absolute path to the light-mode icon
//  
// */

export default function useFavicon(lightHref, darkHref) {
    useEffect(() => {
        const mql = window.matchMedia('(prefers-color-scheme: dark)');

        const update = () => {
            // 1) remove ALL existing favicons
            document.querySelectorAll('link[rel="icon"]').forEach((link) => {
                link.remove();
            }); 

        // 2) create a fresh one
        const link = document.createElement('link');
        link.rel = 'icon';
        link.href = mql.matches ? darkHref : lightHref;
        document.head.appendChild(link);
        };

        update(); // run once now...

        // ... and again whenever the OS theme changes
        mql.addEventListener('change', update);
        return () => {
            mql.removeEventListener('change', update);
            };
    }, [lightHref, darkHref]);
}
