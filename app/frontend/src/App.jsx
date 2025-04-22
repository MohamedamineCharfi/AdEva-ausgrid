import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Dashboard from "./pages/Dashboard";
import ConsumerList from "./pages/ConsumerList";
import Login from "./pages/Login";
import Register from "./pages/Register";
import Settings from "./pages/Settings";
import { ThemeProvider } from "./context/ThemeContext";

function DashboardLayout({ children }) {
  return <div>{children}</div>;
}

export default function App() {
  return (
    <ThemeProvider>
      <div className="transition-colors duration-300 ease-in-out">
        <Router>
          <Routes>
            <Route path="/" element={<DashboardLayout><Dashboard /></DashboardLayout>} />
            <Route path="/consumers" element={<DashboardLayout><ConsumerList /></DashboardLayout>} />
            <Route path="/login" element={<Login />} />
            <Route path="/register" element={<Register />} />
            <Route path="/settings" element={<DashboardLayout><Settings /></DashboardLayout>} />
          </Routes>
        </Router>
      </div>
    </ThemeProvider>
  );
}
