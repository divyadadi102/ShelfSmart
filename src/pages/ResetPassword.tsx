import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { useToast } from "@/hooks/use-toast";
import { Link, useNavigate, useSearchParams } from "react-router-dom";
import { resetPassword } from "@/lib/api";

const ResetPassword = () => {
  const [searchParams] = useSearchParams();
  const token = searchParams.get("token") || "";
  const [newPassword, setNewPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const { toast } = useToast();
  const navigate = useNavigate();

  const validatePassword = (pwd: string) => {
    if (pwd.length < 8) return "Password must be at least 8 characters.";
    if (!/[A-Za-z]/.test(pwd)) return "Password must contain at least one letter.";
    if (!/\d/.test(pwd)) return "Password must contain at least one number.";
    return "";
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const errorMsg = validatePassword(newPassword);
    if (errorMsg) {
      toast({
        title: errorMsg,
        variant: "destructive",
      });
      return;
    }
    if (newPassword !== confirmPassword) {
      toast({
        title: "Passwords do not match",
        variant: "destructive",
      });
      return;
    }
    setIsSubmitting(true);
    try {
      await resetPassword(token, newPassword);
      toast({
        title: "Password reset",
        description: "Your password has been reset. Please login.",
      });
      navigate("/login");
    } catch (err) {
      toast({
        title: "Error",
        description: "Failed to reset password. The link may be invalid or expired.",
        variant: "destructive",
      });
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-indigo-50 to-purple-50 flex items-center justify-center p-4">
      <Card className="w-full max-w-md bg-white/80 backdrop-blur-sm shadow-xl border-0">
        <CardHeader className="text-center">
          <span className="text-2xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
            ShelfSmart
          </span>
          <CardTitle className="text-2xl mt-2">Reset Password</CardTitle>
          <CardDescription>Set a new password for your account</CardDescription>
        </CardHeader>
        <CardContent>
        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="space-y-2">
              <Label htmlFor="new-password">New Password</Label>
              <Input
              id="new-password"
              type="password"
              placeholder="Enter your new password"
              value={newPassword}
              onChange={(e) => setNewPassword(e.target.value)}
              required
              minLength={8}
              />
          </div>
          <div className="space-y-2">
              <Label htmlFor="confirm-password">Confirm Password</Label>
              <Input
              id="confirm-password"
              type="password"
              placeholder="Confirm your new password"
              value={confirmPassword}
              onChange={(e) => setConfirmPassword(e.target.value)}
              required
              minLength={8}
              />
          </div>
          <Button
              type="submit"
              className="w-full bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700"
              disabled={isSubmitting}
          >
              {isSubmitting ? "Resetting..." : "Reset Password"}
          </Button>
          </form>

          <div className="mt-6 text-center">
            <p className="text-sm text-gray-600">
              <Link to="/login" className="text-blue-600 hover:text-blue-700 font-medium">
                Back to Login
              </Link>
            </p>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default ResetPassword;
